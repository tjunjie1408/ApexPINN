"""FastF1 data probe for PLAN T6: availability, missingness, download size, load time, O2 proxy comparison.

Needs network access (run it in Colab or Kaggle, see notebooks/colab/probe_fastf1.ipynb). Example:

    python scripts/probe_fastf1.py --cache-dir /content/drive/MyDrive/apexpinn/.fastf1_cache \
        --json-out /content/drive/MyDrive/apexpinn/experiments/data_probe/probe_results.json \
        --report-out /content/drive/MyDrive/apexpinn/reports/data_probe.md

Races are given as SEASON:GRAND_PRIX:CIRCUIT_ID, where GRAND_PRIX is any name FastF1's event lookup accepts
and CIRCUIT_ID is the id used in configs/decisions_v1.yaml. The O2 comparison uses development-season events
only; holdout-season races contribute availability numbers but never proxy or lap-time statistics.
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
    return pd.DataFrame(
        {"t_s": _seconds(pos, "SessionTime"), "x_m": pos["X"] / 10.0, "y_m": pos["Y"] / 10.0}
    )


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
    keep = ["event_id", "stint_key", "race_lap", "total_laps", "lap_time_s", *probe.PROXY_COLUMNS]
    return summary, scored[keep]


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
    for race in result["races"]:
        if "error" in race:
            continue
        population = f"{race['event_id']} race session"
        for quantity, value in (
            ("FastF1 download size in MB (laps, telemetry, weather, messages); cache-dir size delta",
             race["download_mb"]),
            ("FastF1 session.load wall time in s", race["load_time_s"]),
            ("Share of timed laps with telemetry coverage below 0.5 on car or position stream",
             race["telemetry_missing_fraction"]),
        ):
            records.append(
                MetricRecord(quantity, population, conditions, "scripts/probe_fastf1.py", value, "Observed")
            )
    cmp_cfg = result["proxy_comparison"]
    population = f"scored laps of dev events {cmp_cfg['events']}"
    cmp_conditions = f"phi={cmp_cfg['phi']}, traffic_gap_threshold_s={cmp_cfg['traffic_gap_threshold_s']}"
    for proxy, stats in cmp_cfg["by_proxy"].items():
        for name in ("l2_ratio", "corr_y_within_stint"):
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
    cmp_cfg = result["proxy_comparison"]
    lines += [
        "",
        "## O2 heat-load proxy candidates",
        "",
        f"Scored laps of development events {cmp_cfg['events']} only (holdout races excluded), "
        f"`phi = {cmp_cfg['phi']}`, traffic threshold {cmp_cfg['traffic_gap_threshold_s']} s. "
        "`l2_ratio` = mean within-stint std / std of per-event means (limit L2 is refuted above 0.1). "
        "`corr` = Pearson correlation of stint-centered proxy and stint-centered `y`.",
        "",
        "| Proxy | Laps | Stints | Within-stint std | Across-event std | l2_ratio | corr |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for proxy, s in cmp_cfg["by_proxy"].items():
        lines.append(
            f"| `{proxy}` | {s.get('n_laps', 0)} | {s.get('n_stints', 0)} "
            f"| {_fmt(s.get('within_stint_std'))} | {_fmt(s.get('across_event_std'))} "
            f"| {_fmt(s.get('l2_ratio'))} | {_fmt(s.get('corr_y_within_stint'))} |"
        )
    lines += [
        "",
        f"Mechanical suggestion (largest |corr|, then largest l2_ratio): `{cmp_cfg['suggested']}`. "
        "This is input to the O2 decision, not the decision. With few dev events the across-event std "
        "rests on very few means; add `--race` entries for more development circuits to tighten it.",
        "",
        "## Next steps",
        "",
        "- O1: drop races that are not dry or exceed 20% missing telemetry; extend the probe if needed.",
        "- O2: choose the proxy and normalization from the table above and record the reason.",
        "- O3: check `phi` and the traffic threshold against the traffic shares and slopes above.",
        "- O8: compare pit-lane medians with the `per_circuit_median_iqr` plan.",
        "- O5: user review before any model results are seen.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--race", action="append", type=parse_race, help="SEASON:GRAND_PRIX:CIRCUIT_ID")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".fastf1_cache")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "decisions_v1.yaml")
    parser.add_argument(
        "--json-out", type=Path, default=REPO_ROOT / "experiments" / "data_probe" / "probe_results.json"
    )
    parser.add_argument("--report-out", type=Path, default=REPO_ROOT / "reports" / "data_probe.md")
    args = parser.parse_args(argv)
    races = args.race or [parse_race(s) for s in DEFAULT_RACES]

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
