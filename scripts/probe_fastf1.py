"""FastF1 data probe for PLAN T6: availability, missingness, download size, load time, O2 proxy comparison.

Runs locally and needs network access. FastF1 downloads about 100 MB per race into .fastf1_cache/. Example:

    uv run python scripts/probe_fastf1.py
    uv run python scripts/probe_fastf1.py --preset extended

By default it writes experiments/data_probe/probe_results.json and reports/data_probe.md.

Races are given as SEASON:GRAND_PRIX:CIRCUIT_ID, where GRAND_PRIX is any name FastF1's event lookup accepts
and CIRCUIT_ID is the id used in configs/decisions_v1.yaml. ``--preset default`` probes three races (one per
season); ``--preset extended`` adds eight dry 2023-2024 development races that cover every fold and several
track types, for a tighter O2 comparison. The O2 comparison uses development-season events only;
holdout-season races contribute availability numbers but never proxy or lap-time statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apexpinn.contracts import Decisions, MetricRecord, load_decisions
from apexpinn.data import probe

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RACES = ("2023:Spain:barcelona", "2024:Bahrain:bahrain", "2025:Japan:suzuka")
# Extra development races for --preset extended: dry 2023-2024 races only (never the 2025 holdout), at least
# one per fold, spread over track character. Known wet races (2023 Zandvoort, 2024 Interlagos, 2024 Montreal,
# 2023 Monaco, 2024 Silverstone) are left out; the probe flags wet races anyway.
EXTENDED_EXTRA_RACES = (
    "2024:Japanese Grand Prix:suzuka",  # F1, high-energy
    "2024:Miami Grand Prix:miami",  # F2, hot, semi-street
    "2023:British Grand Prix:silverstone",  # F3, high-energy, fast corners
    "2024:Austrian Grand Prix:spielberg",  # F3, short lap, traction and braking
    "2023:Italian Grand Prix:monza",  # F4, low downforce
    "2023:Hungarian Grand Prix:budapest",  # F4, low grip, slow corners
    "2023:Singapore Grand Prix:singapore",  # F5, street, hot and humid
    "2023:Abu Dhabi Grand Prix:yas_marina",  # F6, dusk race, mixed corners
)
PRESETS = {"default": DEFAULT_RACES, "extended": (*DEFAULT_RACES, *EXTENDED_EXTRA_RACES)}
O1_MAX_TELEMETRY_MISSING = 0.20  # PRD section 9, O1

LAP_COLUMNS = (
    "Driver",
    "DriverNumber",
    "LapTime",
    "LapNumber",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "TrackStatus",
    "IsAccurate",
    "PitInTime",
    "PitOutTime",
    "Deleted",
    "LapStartTime",
    "Time",
    *probe.SPEED_TRAP_COLUMNS,
)
CAR_COLUMNS = ("SessionTime", "Speed", "Throttle", "Brake")
POS_COLUMNS = ("SessionTime", "X", "Y")
WEATHER_COLUMNS = ("Time", "AirTemp", "TrackTemp", "Rainfall")

log = logging.getLogger("probe_fastf1")


def parse_race(spec: str) -> tuple[int, str, str]:
    parts = spec.split(":")
    if len(parts) != 3 or not parts[0].isdigit():
        raise argparse.ArgumentTypeError(f"race {spec!r} must look like SEASON:GRAND_PRIX:CIRCUIT_ID")
    return int(parts[0]), parts[1], parts[2]


def runtime_name() -> str:
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def _seconds(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_timedelta(frame[col]).dt.total_seconds()


def laps_to_frame(laps: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(laps).reset_index(drop=True)
    df["lap_time_s"] = _seconds(df, "LapTime")
    df["lap_end_s"] = _seconds(df, "Time")
    df["lap_start_s"] = _seconds(df, "LapStartTime")
    df["pit_in_s"] = _seconds(df, "PitInTime")
    df["pit_out_s"] = _seconds(df, "PitOutTime")
    return df


def car_to_frame(car: pd.DataFrame | None) -> pd.DataFrame | None:
    if car is None or car.empty:
        return None
    return pd.DataFrame(
        {
            "t_s": _seconds(car, "SessionTime"),
            "speed_kmh": car["Speed"].astype(float),
            "brake": car["Brake"],
        }
    )


def pos_to_frame(pos: pd.DataFrame | None) -> pd.DataFrame | None:
    if pos is None or pos.empty:
        return None
    # FastF1 positions are in 1/10 m.
    return pd.DataFrame({"t_s": _seconds(pos, "SessionTime"), "x_m": pos["X"] / 10.0, "y_m": pos["Y"] / 10.0})


def _safe_attr(session: Any, name: str) -> Any:
    """Session attribute, or None if FastF1 could not load it (it raises DataNotLoadedError then)."""
    try:
        return getattr(session, name)
    except Exception:  # FastF1 raises its own exception types for data it failed to load
        log.warning("session.%s is not available", name)
        return None


def _concat_report(frames: dict[str, pd.DataFrame], columns: tuple[str, ...]) -> dict[str, Any]:
    if not frames:
        return probe.column_report(None, columns)
    return probe.column_report(pd.concat(frames.values(), ignore_index=True), columns)


def probe_race(
    season: int, grand_prix: str, circuit: str, cache_dir: Path, decisions: Decisions
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Load one race and return (summary, scored-lap frame with proxies or None)."""
    import fastf1

    event_id = f"{season}_{circuit}"
    if event_id in decisions.dev_events:
        role = "dev"
    elif event_id in decisions.holdout_events:
        role = "holdout"
    else:
        role = "not_in_config"
    summary: dict[str, Any] = {
        "event_id": event_id,
        "season": season,
        "grand_prix_query": grand_prix,
        "circuit": circuit,
        "fold": decisions.circuit_to_fold.get(circuit),
        "role": role,
    }
    size_before = probe.dir_size_bytes(cache_dir)
    t0 = time.perf_counter()
    session = fastf1.get_session(season, grand_prix, "R")
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    summary["load_time_s"] = time.perf_counter() - t0
    summary["download_mb"] = (probe.dir_size_bytes(cache_dir) - size_before) / 1e6
    summary["cache_dir_mb_after"] = probe.dir_size_bytes(cache_dir) / 1e6
    summary["event_name"] = str(session.event.get("EventName", ""))
    summary["event_date"] = str(session.event.get("EventDate", ""))
    summary["location"] = str(session.event.get("Location", ""))

    laps = laps_to_frame(session.laps)
    total_laps = _safe_attr(session, "total_laps") or (int(laps["LapNumber"].max()) if len(laps) else None)
    summary["total_laps"] = total_laps
    summary["n_drivers"] = int(laps["Driver"].nunique()) if "Driver" in laps else 0
    summary["n_laps"] = len(laps)

    car_frames = {str(k): v for k, v in (_safe_attr(session, "car_data") or {}).items()}
    pos_frames = {str(k): v for k, v in (_safe_attr(session, "pos_data") or {}).items()}
    weather = _safe_attr(session, "weather_data")
    summary["columns"] = {
        "laps": probe.column_report(laps, LAP_COLUMNS),
        "car": _concat_report(car_frames, CAR_COLUMNS),
        "pos": _concat_report(pos_frames, POS_COLUMNS),
        "weather": probe.column_report(weather, WEATHER_COLUMNS),
    }
    summary["drivers_without_car_data"] = sorted(set(laps["DriverNumber"].astype(str)) - set(car_frames))
    summary["drivers_without_pos_data"] = sorted(set(laps["DriverNumber"].astype(str)) - set(pos_frames))

    if weather is not None and not weather.empty:
        if "Rainfall" in weather:
            summary["rainfall_any"] = bool(weather["Rainfall"].eq(True).any())
        for col in ("TrackTemp", "AirTemp"):
            if col in weather:
                summary[f"{col.lower()}_c"] = {
                    "min": float(weather[col].min()),
                    "mean": float(weather[col].mean()),
                    "max": float(weather[col].max()),
                }
    compounds = laps["Compound"].astype(str).str.upper() if "Compound" in laps else pd.Series(dtype=str)
    wet = compounds.isin(probe.WET_COMPOUNDS)
    summary["wet_lap_fraction"] = float(wet.mean()) if len(compounds) else None
    summary["is_dry"] = not summary.get("rainfall_any") and not summary["wet_lap_fraction"]

    # Per-lap telemetry aggregates, per driver, from that lap's samples only.
    tele_parts = []
    for number, group in laps.groupby(laps["DriverNumber"].astype(str)):
        part = probe.per_lap_telemetry(
            group, car_to_frame(car_frames.get(number)), pos_to_frame(pos_frames.get(number))
        )
        tele_parts.append(part)
    tele = pd.concat(tele_parts) if tele_parts else pd.DataFrame(index=laps.index)
    laps = laps.join(tele)
    timed = laps["lap_time_s"].notna()
    summary["telemetry_missing_fraction"] = float(laps.loc[timed, "telemetry_missing"].mean())
    summary["o1_keep"] = bool(
        summary["is_dry"] and summary["telemetry_missing_fraction"] <= O1_MAX_TELEMETRY_MISSING
    )

    laps["gap_ahead_s"] = probe.gap_to_car_ahead(laps)
    laps["exclusion_reason"] = probe.exclusion_reason(laps, decisions.traffic_gap_threshold_s)
    counts = laps["exclusion_reason"].replace("", "scored").value_counts()
    summary["exclusion_counts"] = {str(k): int(v) for k, v in counts.items()}

    if summary["role"] != "dev":
        # Holdout races give availability numbers only. Traffic, pit-lane, proxy, and lap-time statistics
        # inform O2, O3, and O8, so they come from development races alone.
        return summary, None

    green_timed = timed & laps["TrackStatus"].map(probe.is_green)
    summary["traffic_share_green_laps"] = {
        str(thr): float(laps.loc[green_timed, "gap_ahead_s"].lt(thr).mean()) for thr in (0.5, 1.0, 1.5, 2.0)
    }
    pit = probe.pit_lane_times(laps)
    summary["pit_lane_time_s"] = {
        "n": int(pit.size),
        "median": float(np.median(pit)) if pit.size else None,
        "q25": float(np.percentile(pit, 25)) if pit.size else None,
        "q75": float(np.percentile(pit, 75)) if pit.size else None,
    }

    # Stint lap = 1-based position in the driver's stint over all laps before exclusions, so the out-lap
    # (or race lap 1) is stint lap 1, as in the synthetic generator. TyreLife is kept for comparison; it
    # starts above 1 on sets used earlier in the weekend.
    stint_start = laps.groupby(["Driver", "Stint"])["LapNumber"].transform("min")
    laps["stint_lap"] = laps["LapNumber"] - stint_start + 1
    scored = laps[(laps["exclusion_reason"] == "") & ~laps["telemetry_missing"]].copy()
    scored["event_id"] = event_id
    scored["stint_key"] = event_id + "/" + scored["Driver"].astype(str) + "/" + scored["Stint"].astype(str)
    scored["race_lap"] = scored["LapNumber"].astype(float)
    scored["total_laps"] = float(total_laps) if total_laps else np.nan
    scored["speed_trap"] = probe.speed_trap_proxy(scored)
    slopes = probe.stint_slopes(scored)
    summary["raw_stint_slope_s_per_lap"] = {
        "n_stints": int(slopes.size),
        "median": float(np.median(slopes)) if slopes.size else None,
    }
    summary["warmup_visibility"] = probe.warmup_visibility(scored, decisions.phi_mid)
    keep = [
        "event_id",
        "stint_key",
        "stint_lap",
        "TyreLife",
        "race_lap",
        "total_laps",
        "lap_time_s",
        *probe.PROXY_COLUMNS,
    ]
    return summary, scored.reindex(columns=keep)


def environment() -> dict[str, Any]:
    import fastf1

    return {
        "runtime": runtime_name(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "fastf1": fastf1.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def build_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    env = result["environment"]
    conditions = (
        f"runtime={env['runtime']}, fastf1={env['fastf1']}; download is the cache-dir size delta, "
        "so 0 MB means the cache was already warm"
    )
    records = []
    phi = result["proxy_comparison"]["phi"]
    for race in result["races"]:
        if "error" in race:
            continue
        population = f"{race['event_id']} race session"
        for quantity, value in (
            (
                "FastF1 download size in MB (laps, telemetry, weather, messages); cache-dir size delta",
                race["download_mb"],
            ),
            ("FastF1 session.load wall time in s", race["load_time_s"]),
            (
                "Share of timed laps with telemetry coverage below 0.5 on car or position stream",
                race["telemetry_missing_fraction"],
            ),
        ):
            records.append(
                MetricRecord(quantity, population, conditions, "scripts/probe_fastf1.py", value, "Observed")
            )
        if "warmup_visibility" in race:
            warm = race["warmup_visibility"]
            warm_population = f"{race['event_id']} scored laps, stints with at least 3 scored laps"
            warm_conditions = (
                f"phi={phi}, y = lap_time - phi*(L - race_lap); dip = fastest scored lap strictly after the "
                f"first scored lap, at stint lap 2-8, more than {warm['min_gain_s']} s faster than the first "
                "scored lap, and last scored lap slower; stint lap = position in stint"
            )
            for quantity, value in (
                ("Share of stints with an early corrected-pace dip (S4 on real data)", warm["dip_share"]),
                (
                    "Median gain in s of the fastest lap over the first scored lap, dip stints",
                    warm["dip_gain_s"]["median"],
                ),
                (
                    "Median gain in s of the fastest lap over the first scored lap, all stints",
                    warm["gain_all_s"]["median"],
                ),
            ):
                records.append(
                    MetricRecord(
                        quantity,
                        warm_population,
                        warm_conditions,
                        "apexpinn.data.probe.warmup_visibility",
                        value,
                        "Observed",
                    )
                )
    cmp_cfg = result["proxy_comparison"]
    population = f"scored laps of dev events {cmp_cfg['events']}"
    cmp_conditions = (
        f"phi={cmp_cfg['phi']}, traffic_gap_threshold_s={cmp_cfg['traffic_gap_threshold_s']}, "
        f"position smoothing window={cmp_cfg['smooth_window']} samples"
    )
    for proxy, stats in cmp_cfg["by_proxy"].items():
        for name in (
            "between_event_share",
            "l2_ratio",
            "noise_std",
            "corr_y_within_stint",
            "partial_corr_y_given_lap",
        ):
            if stats.get(name) is None:
                continue
            records.append(
                MetricRecord(
                    f"{proxy} {name}",
                    population,
                    cmp_conditions,
                    "apexpinn.data.probe.compare_proxies",
                    stats[name],
                    "Observed",
                )
            )
    return [asdict(r) for r in records]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(result: dict[str, Any]) -> str:
    env = result["environment"]
    ok = [r for r in result["races"] if "error" not in r]
    lines = [
        "# FastF1 data probe (PLAN T6)",
        "",
        f"Generated {result['generated_utc']} by `scripts/probe_fastf1.py` on {env['runtime']} "
        f"(Python {env['python']}, FastF1 {env['fastf1']}, pandas {env['pandas']}). "
        "All numbers are `Observed` on the races below.",
        "",
        "## Races",
        "",
        "| Event | Role | Fold | Load s | Download MB | Laps | Telemetry missing | Dry | O1 keep |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |",
    ]
    for r in ok:
        lines.append(
            f"| {r['event_id']} ({r['event_name']}) | {r['role']} | {r['fold']} "
            f"| {_fmt(r['load_time_s'], 1)} | {_fmt(r['download_mb'], 1)} | {r['n_laps']} "
            f"| {_fmt(r['telemetry_missing_fraction'])} | {r['is_dry']} | {r['o1_keep']} |"
        )
    for r in result["races"]:
        if "error" in r:
            lines.append(f"| {r['event_id']} | error: {r['error']} | | | | | | | |")
    proj = result["projection"]
    lines += [
        "",
        f"**E7 projection**: mean download {_fmt(proj['mean_download_mb'], 1)} MB and "
        f"{_fmt(proj['mean_load_time_s'], 1)} s per race; {proj['n_config_events']} events in the config "
        f"give about {_fmt(proj['projected_total_gb'], 2)} GB and {_fmt(proj['projected_total_h'], 2)} h. "
        "A download of 0 MB means the cache was already warm for that race.",
        "",
        "## Column availability (missing fraction)",
        "",
    ]
    for source in ("laps", "car", "pos", "weather"):
        lines += [f"### {source}", "", "| Column | " + " | ".join(r["event_id"] for r in ok) + " |"]
        lines.append("| :--- |" + " ---: |" * len(ok))
        columns = ok[0]["columns"][source].keys() if ok else []
        for col in columns:
            cells = []
            for r in ok:
                info = r["columns"][source][col]
                cells.append("absent" if not info["present"] else _fmt(info["nan_fraction"]))
            lines.append(f"| `{col}` | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Exclusions, traffic, pit lane",
        "",
        "Traffic shares, pit-lane times, and stint slopes come from development races only; holdout races "
        "report exclusion counts and availability.",
        "",
    ]
    for r in ok:
        line = f"- **{r['event_id']}** ({r['role']}): exclusions {r['exclusion_counts']}."
        if "traffic_share_green_laps" in r:
            traffic = ", ".join(f"<{k} s: {_fmt(v)}" for k, v in r["traffic_share_green_laps"].items())
            line += f" Share of green timed laps with gap ahead {traffic}."
        if "pit_lane_time_s" in r:
            pit = r["pit_lane_time_s"]
            line += (
                f" Pit-lane time median {_fmt(pit['median'], 2)} s, IQR "
                f"[{_fmt(pit['q25'], 2)}, {_fmt(pit['q75'], 2)}] s over {pit['n']} stops."
            )
        if "raw_stint_slope_s_per_lap" in r:
            s = r["raw_stint_slope_s_per_lap"]
            line += f" Median raw within-stint slope {_fmt(s['median'])} s/lap over {s['n_stints']} stints."
        line += f" Drivers without car data: {r['drivers_without_car_data'] or 'none'}."
        lines.append(line)
    warm_races = [r for r in ok if "warmup_visibility" in r]
    if warm_races:
        margin = warm_races[0]["warmup_visibility"]["min_gain_s"]
        lines += [
            "",
            "## Warm-up visibility (S4 on real stints)",
            "",
            "Development races only, scored laps after all exclusions, stints with at least 3 scored laps. "
            "`y = lap_time - phi_mid * (L - race_lap)`. Stint lap is the 1-based position in the stint over "
            "all laps before exclusions, so stint lap 1 is the out-lap (or race lap 1) and is normally "
            "excluded. Gain is `y` of the first scored lap minus the minimum `y` of the stint. A dip means "
            "the fastest scored lap comes strictly after the first scored lap, at stint lap 2-8, with a gain "
            f"above {_fmt(margin, 2)} s, and the last scored lap is slower (PRD S4, PLAN T3 test 3). A stint "
            "whose first scored lap is its fastest is plain wear and never counts as a dip. The all-stint "
            "gain columns show how much early improvement exists regardless of the dip rule.",
            "",
            "| Event | Stints | Dip stints | Dip share | Dip gain median s | Dip gain IQR s "
            f"| All-stint gain median s | All-stint gain IQR s | Share gain > {_fmt(margin, 2)} s "
            "| First scored stint lap (median; counts) | First scored TyreLife median |",
            "| :--- | ---: | ---: | ---: | ---: | :--- | ---: | :--- | ---: | :--- | ---: |",
        ]
        for r in warm_races:
            w = r["warmup_visibility"]
            gain, every, first = w["dip_gain_s"], w["gain_all_s"], w["first_scored_stint_lap"]
            counts = ", ".join(f"{k}: {v}" for k, v in first["counts"].items()) or "none"
            lines.append(
                f"| {r['event_id']} | {w['n_stints']} | {w['n_dip']} | {_fmt(w['dip_share'])} "
                f"| {_fmt(gain['median'])} | [{_fmt(gain['q25'])}, {_fmt(gain['q75'])}] "
                f"| {_fmt(every['median'])} | [{_fmt(every['q25'])}, {_fmt(every['q75'])}] "
                f"| {_fmt(every['share_above_min_gain'])} "
                f"| {_fmt(first['median'], 1)}; {counts} | {_fmt(w['first_scored_tyre_life_median'], 1)} |"
            )
    cmp_cfg = result["proxy_comparison"]
    lines += [
        "",
        "## O2 heat-load proxy candidates",
        "",
        f"Scored laps of development events {cmp_cfg['events']} only (holdout races excluded), "
        f"`phi = {cmp_cfg['phi']}`, traffic threshold {cmp_cfg['traffic_gap_threshold_s']} s. "
        "`energy_index` uses positions smoothed within each lap (Savitzky-Golay, "
        f"{cmp_cfg['smooth_window']} samples, cubic) and drops samples above {_fmt(probe.MAX_ACCEL_MS2, 0)} "
        "m/s^2 as glitches.",
        "",
        "Columns, all over stints with at least 3 scored laps:",
        "",
        "- `between share`: share of the proxy's variance over laps that lies between event means (an "
        "ICC-like ratio, 0 to 1). A circuit-level heat index should be high here.",
        "- `within std`, `across std`, `l2_ratio`: mean within-stint std, std of per-event means, and their "
        "ratio (limit L2 is refuted above 0.1).",
        "- `noise std`: mean within-stint std after removing a linear race-lap trend. `drift/lap`: pooled "
        "within-stint slope per race lap.",
        "- `corr`: Pearson correlation of stint-centered proxy and stint-centered `y`. `partial corr`: the "
        "same after removing a linear race-lap trend from both. These are leakage warnings, not reasons to "
        "pick a proxy: a proxy that moves with lap time (trap speed is higher on faster laps, a slow lap "
        "holds more braking time) feeds the target back into the model input. `leak?` is set when |corr| "
        f"or |partial corr| is above {probe.LEAK_WARNING_ABS_CORR}.",
        "",
        "| Proxy | Laps | Stints | Events | between share | within std | across std | l2_ratio "
        "| noise std | drift/lap | corr | partial corr | leak? | score |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |",
    ]
    for proxy, s in cmp_cfg["by_proxy"].items():
        lines.append(
            f"| `{proxy}` | {s.get('n_laps', 0)} | {s.get('n_stints', 0)} | {s.get('n_events', 0)} "
            f"| {_fmt(s.get('between_event_share'))} | {_fmt(s.get('within_stint_std'))} "
            f"| {_fmt(s.get('across_event_std'))} | {_fmt(s.get('l2_ratio'))} "
            f"| {_fmt(s.get('noise_std'))} | {_fmt(s.get('drift_per_lap'), 4)} "
            f"| {_fmt(s.get('corr_y_within_stint'))} | {_fmt(s.get('partial_corr_y_given_lap'))} "
            f"| {'yes' if s.get('leak_warning') else 'no'} | {_fmt(s.get('suggest_score'))} |"
        )
    lines += [
        "",
        f"Mechanical suggestion: `{cmp_cfg['suggested']}`. Rule: largest `score = between share x (1 - "
        "|corr|)`, so proxies are ranked by circuit-level separation and discounted for tracking lap time. "
        "This is input to the O2 decision, not the decision. With few dev events the between-event "
        "statistics rest on very few means; run `--preset extended` (or add `--race` entries) to tighten "
        "them.",
        "",
        "## Next steps",
        "",
        "- O1: drop races that are not dry or exceed 20% missing telemetry; extend the probe if needed.",
        "- O2: choose the proxy and normalization from the table above and record the reason.",
        "- O3: check `phi` and the traffic threshold against the traffic shares and slopes above.",
        "- O8: compare pit-lane medians with the `per_circuit_median_iqr` plan.",
        "- T5: use the warm-up visibility table to judge whether real stints show the S4 dip before fitting.",
        "- O5: user review before any model results are seen.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    races_group = parser.add_mutually_exclusive_group()
    races_group.add_argument(
        "--race", action="append", type=parse_race, help="SEASON:GRAND_PRIX:CIRCUIT_ID (repeatable)"
    )
    races_group.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help="named race list: default (3 races) or extended (default plus 8 dev races)",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".fastf1_cache")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "decisions_v1.yaml")
    parser.add_argument(
        "--json-out", type=Path, default=REPO_ROOT / "experiments" / "data_probe" / "probe_results.json"
    )
    parser.add_argument("--report-out", type=Path, default=REPO_ROOT / "reports" / "data_probe.md")
    args = parser.parse_args(argv)
    races = args.race or [parse_race(s) for s in PRESETS[args.preset]]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import fastf1

    fastf1.set_log_level("WARNING")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(args.cache_dir))
    decisions = load_decisions(args.config)

    summaries: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    for season, grand_prix, circuit in races:
        log.info("probing %s %s (%s)", season, grand_prix, circuit)
        try:
            summary, scored = probe_race(season, grand_prix, circuit, args.cache_dir, decisions)
        except Exception as exc:  # one bad race must not lose the others
            log.exception("race %s %s failed", season, grand_prix)
            summary, scored = {"event_id": f"{season}_{circuit}", "error": repr(exc)}, None
        summaries.append(summary)
        if scored is not None:
            scored_frames.append(scored)

    scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    by_proxy = probe.compare_proxies(scored_all, decisions.phi_mid) if len(scored_all) else {}
    ok = [s for s in summaries if "error" not in s]
    n_config = len(decisions.dev_events) + len(decisions.holdout_events)
    mean_mb = float(np.mean([s["download_mb"] for s in ok])) if ok else float("nan")
    mean_s = float(np.mean([s["load_time_s"] for s in ok])) if ok else float("nan")
    result = {
        "schema": "apexpinn.data_probe/v1",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": environment(),
        "config": str(args.config),
        "preset": "custom" if args.race else args.preset,
        "races": summaries,
        "projection": {
            "n_config_events": n_config,
            "mean_download_mb": mean_mb,
            "mean_load_time_s": mean_s,
            "projected_total_gb": mean_mb * n_config / 1e3,
            "projected_total_h": mean_s * n_config / 3600,
        },
        "proxy_comparison": {
            "events": sorted(scored_all["event_id"].unique().tolist()) if len(scored_all) else [],
            "phi": decisions.phi_mid,
            "traffic_gap_threshold_s": decisions.traffic_gap_threshold_s,
            "smooth_window": probe.SMOOTH_WINDOW,
            "by_proxy": by_proxy,
            "suggested": probe.suggest_proxy(by_proxy),
        },
    }
    result = probe.to_jsonable(result)
    result["records"] = probe.to_jsonable(build_records(result))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(result), encoding="utf-8")
    log.info("wrote %s and %s", args.json_out, args.report_out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
