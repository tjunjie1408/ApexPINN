"""Pure helpers for the FastF1 data probe (PLAN T6). No network and no FastF1 import here.

The CLI in scripts/probe_fastf1.py loads sessions and converts them to the plain frames used below:

- laps: one row per lap with float-second columns (see ``laps_to_frame`` in the script).
- car: ``t_s`` (session time, s), ``speed_kmh``, ``brake`` (bool).
- pos: ``t_s`` (session time, s), ``x_m``, ``y_m``.

Results feed reports/data_probe.md and the O1-O8 freeze. They are probe statistics, not the Silver/Gold
pipeline (T8-T10), which will define the official filters and proxies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WET_COMPOUNDS = ("INTERMEDIATE", "WET")
SPEED_TRAP_COLUMNS = ("SpeedI1", "SpeedI2", "SpeedFL", "SpeedST")
PROXY_COLUMNS = ("speed_trap", "energy_index", "brake_fraction_weighted")
MIN_SPEED_MS = 5.0  # below this the heading from positions is noise (pit lane, grid)
MIN_STEP_M = 0.5  # minimum distance between position samples for a usable heading


def column_report(frame: pd.DataFrame | None, expected: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Per expected column: whether it exists and its missing-value fraction (None if absent)."""
    out: dict[str, dict[str, Any]] = {}
    n = 0 if frame is None else len(frame)
    for col in expected:
        if frame is None or col not in frame.columns:
            out[col] = {"present": False, "nan_fraction": None}
        else:
            frac = float(frame[col].isna().mean()) if n else None
            out[col] = {"present": True, "nan_fraction": frac}
    return out


def dir_size_bytes(path: str | Path) -> int:
    """Total size of all files below ``path`` (0 if it does not exist)."""
    root = Path(path)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def is_green(track_status: Any) -> bool:
    """FastF1 TrackStatus is a string of status codes seen during the lap. Green means only '1'."""
    if not isinstance(track_status, str) or not track_status:
        return False
    return set(track_status) <= {"1"}


def gap_to_car_ahead(laps: pd.DataFrame) -> pd.Series:
    """Gap in s at the timing line to the car that crossed it just before on the same lap number.

    Uses ``lap_end_s`` (session time at the end of the lap). The leader of each lap gets NaN. Cars a lap
    down are not counted as the car ahead, so traffic from lapped cars is under-counted.
    """
    gaps = pd.Series(np.nan, index=laps.index, dtype=float)
    for _, group in laps.groupby("LapNumber"):
        ordered = group["lap_end_s"].dropna().sort_values()
        gaps.loc[ordered.index] = ordered.diff().to_numpy()
    return gaps


def exclusion_reason(laps: pd.DataFrame, traffic_gap_threshold_s: float) -> pd.Series:
    """First matching PROJECT section 4.5 filter per lap, or "" for a scored lap.

    Expected columns: LapNumber, lap_time_s, pit_in_s, pit_out_s, TrackStatus, IsAccurate, Deleted,
    Compound, gap_ahead_s. Missing optional columns are treated as "filter not triggered".
    """

    def col(name: str, default: Any) -> pd.Series:
        if name in laps.columns:
            return laps[name]
        return pd.Series(default, index=laps.index)

    rules: list[tuple[str, pd.Series]] = [
        ("no_lap_time", col("lap_time_s", np.nan).isna()),
        ("race_lap_1", col("LapNumber", np.nan) == 1),
        ("in_lap", col("pit_in_s", np.nan).notna()),
        ("out_lap", col("pit_out_s", np.nan).notna()),
        ("not_green", ~col("TrackStatus", "1").map(is_green).astype(bool)),
        ("inaccurate", ~col("IsAccurate", True).eq(True)),  # missing counts as inaccurate
        ("deleted", col("Deleted", False).eq(True)),
        ("wet_compound", col("Compound", "").astype(str).str.upper().isin(WET_COMPOUNDS)),
        ("traffic", col("gap_ahead_s", np.nan).lt(traffic_gap_threshold_s)),
    ]
    reason = pd.Series("", index=laps.index, dtype=object)
    for name, mask in rules:
        reason = reason.mask((reason == "") & mask.to_numpy(dtype=bool), name)
    return reason


def pit_lane_times(laps: pd.DataFrame, low_s: float = 5.0, high_s: float = 60.0) -> np.ndarray:
    """Pit-lane time per stop: next lap's ``pit_out_s`` minus the in-lap's ``pit_in_s`` (same driver).

    Values outside ``[low_s, high_s]`` (red flags, retirements, garage stops) are dropped.
    """
    values: list[float] = []
    for _, group in laps.sort_values("LapNumber").groupby("Driver"):
        pit_in = group["pit_in_s"].to_numpy(dtype=float)
        pit_out = group["pit_out_s"].to_numpy(dtype=float)
        lap_no = group["LapNumber"].to_numpy(dtype=float)
        for i in range(len(group) - 1):
            if np.isfinite(pit_in[i]) and np.isfinite(pit_out[i + 1]) and lap_no[i + 1] == lap_no[i] + 1:
                values.append(pit_out[i + 1] - pit_in[i])
    arr = np.asarray(values, dtype=float)
    return arr[(arr >= low_s) & (arr <= high_s)]


def lateral_accel(t_s: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, v_ms: np.ndarray) -> np.ndarray:
    """|a_lat| = v * |d(heading)/dt| from planar positions and speed, m/s^2.

    NaN where the car is slow or barely moves between samples, because the heading is undefined there.
    Uses only the given samples: central differences inside, second-order one-sided differences at the
    two ends, so a single lap's slice gives an estimate that does not depend on neighbouring laps.
    """
    t = np.asarray(t_s, dtype=float)
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    if t.size < 3:
        return np.full(t.shape, np.nan)
    dx = np.gradient(x, edge_order=2)
    dy = np.gradient(y, edge_order=2)
    heading = np.unwrap(np.arctan2(dy, dx))
    yaw_rate = np.gradient(heading, t, edge_order=2)
    a_lat = np.abs(v * yaw_rate)
    unusable = (v < MIN_SPEED_MS) | (np.hypot(dx, dy) < MIN_STEP_M) | ~np.isfinite(a_lat)
    a_lat[unusable] = np.nan
    return a_lat


def longitudinal_accel(t_s: np.ndarray, v_ms: np.ndarray) -> np.ndarray:
    """|dv/dt| from the speed channel, m/s^2."""
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    if t.size < 2:
        return np.full(t.shape, np.nan)
    return np.abs(np.gradient(v, t))


def _dedupe_sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=["t_s"]).sort_values("t_s").drop_duplicates("t_s").reset_index(drop=True)


def _rate_hz(t: np.ndarray) -> float:
    if t.size < 2 or t[-1] <= t[0]:
        return float("nan")
    return (t.size - 1) / (t[-1] - t[0])


def _lap_energy(
    t_car: np.ndarray, v_car: np.ndarray, t_pos: np.ndarray, x_m: np.ndarray, y_m: np.ndarray
) -> np.ndarray:
    """|a_lat| + |a_lon| on one lap's position samples, from that lap's car and position samples only.

    Derivatives use one-sided differences at the lap edges, so no sample of the previous or next lap
    enters (PROJECT section 5.1). Speed and |a_lon| are interpolated within the lap's car samples.
    """
    if t_pos.size < 3 or t_car.size < 2:
        return np.full(t_pos.shape, np.nan)
    v_pos = np.interp(t_pos, t_car, v_car)
    a_lat = lateral_accel(t_pos, x_m, y_m, v_pos)
    a_lon = np.interp(t_pos, t_car, longitudinal_accel(t_car, v_car))
    return a_lat + a_lon


def per_lap_telemetry(
    laps: pd.DataFrame, car: pd.DataFrame | None, pos: pd.DataFrame | None, min_coverage: float = 0.5
) -> pd.DataFrame:
    """Per-lap telemetry aggregates for one driver, using only samples inside each lap.

    ``laps`` needs ``lap_start_s`` and ``lap_end_s``. Returns a frame aligned to ``laps.index`` with
    sample counts, coverage (samples / expected samples at the stream's average rate for this driver), a
    ``telemetry_missing`` flag (coverage below ``min_coverage`` on either stream), and the proxies
    ``energy_index`` (mean |a_lat| + |a_lon| on position samples, m/s^2), ``brake_fraction``, and
    ``brake_fraction_weighted`` (v^2-weighted share of braking samples). Derivatives for ``energy_index``
    are taken within each lap's samples, so changing a later lap's telemetry never changes an earlier lap.
    """
    cols = [
        "n_car",
        "n_pos",
        "car_coverage",
        "pos_coverage",
        "telemetry_missing",
        "speed_mean_kmh",
        "brake_fraction",
        "brake_fraction_weighted",
        "energy_index",
    ]
    start = laps["lap_start_s"].to_numpy(dtype=float)
    end = laps["lap_end_s"].to_numpy(dtype=float)
    duration = end - start

    empty_car = car is None or car.empty
    empty_pos = pos is None or pos.empty
    car = pd.DataFrame(columns=["t_s", "speed_kmh", "brake"]) if empty_car else _dedupe_sorted(car)
    pos = pd.DataFrame(columns=["t_s", "x_m", "y_m"]) if empty_pos else _dedupe_sorted(pos)

    t_car = car["t_s"].to_numpy(dtype=float)
    v_car = car["speed_kmh"].to_numpy(dtype=float) / 3.6
    brake = car["brake"].eq(True).to_numpy(dtype=bool)
    t_pos = pos["t_s"].to_numpy(dtype=float)
    x_pos, y_pos = pos["x_m"].to_numpy(dtype=float), pos["y_m"].to_numpy(dtype=float)

    rate_car, rate_pos = _rate_hz(t_car), _rate_hz(t_pos)
    ok = np.isfinite(start) & np.isfinite(end) & (end > start)
    car_lo, car_hi = np.searchsorted(t_car, start), np.searchsorted(t_car, end)
    pos_lo, pos_hi = np.searchsorted(t_pos, start), np.searchsorted(t_pos, end)

    rows = []
    for i in range(len(laps)):
        if not ok[i]:
            rows.append([0, 0, 0.0, 0.0, True] + [np.nan] * 4)
            continue
        cs = slice(car_lo[i], car_hi[i])
        ps = slice(pos_lo[i], pos_hi[i])
        n_car, n_pos = cs.stop - cs.start, ps.stop - ps.start
        cov_car = n_car / (duration[i] * rate_car) if np.isfinite(rate_car) else 0.0
        cov_pos = n_pos / (duration[i] * rate_pos) if np.isfinite(rate_pos) else 0.0
        missing = bool(cov_car < min_coverage or cov_pos < min_coverage)
        if n_car:
            v2 = v_car[cs] ** 2
            speed_mean = float(np.mean(v_car[cs]) * 3.6)
            brake_frac = float(np.mean(brake[cs]))
            brake_w = float(np.sum(v2 * brake[cs]) / np.sum(v2)) if np.sum(v2) > 0 else np.nan
        else:
            speed_mean = brake_frac = brake_w = np.nan
        lap_energy = _lap_energy(t_car[cs], v_car[cs], t_pos[ps], x_pos[ps], y_pos[ps])
        energy_idx = float(np.nanmean(lap_energy)) if np.isfinite(lap_energy).any() else np.nan
        rows.append([n_car, n_pos, cov_car, cov_pos, missing, speed_mean, brake_frac, brake_w, energy_idx])

    out = pd.DataFrame(rows, index=laps.index, columns=cols)
    out["telemetry_missing"] = out["telemetry_missing"].astype(bool)
    return out


def speed_trap_proxy(laps: pd.DataFrame) -> pd.Series:
    """Mean of the available speed-trap columns (km/h) per lap."""
    present = [c for c in SPEED_TRAP_COLUMNS if c in laps.columns]
    if not present:
        return pd.Series(np.nan, index=laps.index)
    return laps[present].astype(float).mean(axis=1, skipna=True)


def compare_proxies(
    frame: pd.DataFrame, phi: float, proxies: Sequence[str] = PROXY_COLUMNS, min_stint_laps: int = 3
) -> dict[str, Any]:
    """Compare heat-load proxy candidates on scored laps (PRD section 9, O2).

    ``frame`` needs ``event_id``, ``stint_key``, ``race_lap``, ``total_laps``, ``lap_time_s`` and the proxy
    columns. For each proxy:

    - ``within_stint_std``: mean over stints (at least ``min_stint_laps`` laps) of the proxy std.
    - ``across_event_std``: std (ddof=1) of the per-event proxy means. NaN with fewer than 2 events.
    - ``l2_ratio``: within_stint_std / across_event_std. Limit L2 is refuted if this is above 0.1.
    - ``corr_y_within_stint``: Pearson correlation of stint-centered proxy and stint-centered
      ``y = lap_time_s - phi * (total_laps - race_lap)``.
    """
    y = frame["lap_time_s"] - phi * (frame["total_laps"] - frame["race_lap"])
    results: dict[str, Any] = {}
    for proxy in proxies:
        if proxy not in frame.columns:
            results[proxy] = {"available": False}
            continue
        sub = pd.DataFrame(
            {"event_id": frame["event_id"], "stint": frame["stint_key"], "q": frame[proxy], "y": y}
        ).dropna()
        sizes = sub.groupby("stint")["q"].transform("size")
        sub = sub[sizes >= min_stint_laps]
        if sub.empty:
            results[proxy] = {"available": True, "n_laps": 0}
            continue
        stint_std = sub.groupby("stint")["q"].std(ddof=0)
        event_means = sub.groupby("event_id")["q"].mean()
        across = float(event_means.std(ddof=1)) if len(event_means) >= 2 else float("nan")
        within = float(stint_std.mean())
        q_c = sub["q"] - sub.groupby("stint")["q"].transform("mean")
        y_c = sub["y"] - sub.groupby("stint")["y"].transform("mean")
        denom = float(np.sqrt((q_c**2).sum() * (y_c**2).sum()))
        corr = float((q_c * y_c).sum() / denom) if denom > 0 else float("nan")
        results[proxy] = {
            "available": True,
            "n_laps": int(len(sub)),
            "n_stints": int(sub["stint"].nunique()),
            "n_events": int(len(event_means)),
            "within_stint_std": within,
            "across_event_std": across,
            "l2_ratio": within / across if across and np.isfinite(across) else float("nan"),
            "corr_y_within_stint": corr,
            "event_means": {str(k): float(v) for k, v in event_means.items()},
        }
    return results


def suggest_proxy(comparison: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Suggestion only: largest |corr_y_within_stint|, ties broken by larger l2_ratio.

    The user makes the O2 decision.
    """

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float]:
        stats = item[1]
        corr = stats.get("corr_y_within_stint", float("nan"))
        ratio = stats.get("l2_ratio", float("nan"))
        return (
            abs(corr) if np.isfinite(corr) else -1.0,
            ratio if np.isfinite(ratio) else -1.0,
        )

    usable = [(k, v) for k, v in comparison.items() if v.get("available") and v.get("n_laps")]
    if not usable:
        return None
    return max(usable, key=key)[0]


def stint_slopes(frame: pd.DataFrame, min_laps: int = 5) -> np.ndarray:
    """Raw lap-time slope (s per race lap) of each stint with at least ``min_laps`` scored laps.

    Descriptive only for O3: within a stint the raw slope mixes wear and the race-lap trend phi.
    """
    slopes = []
    for _, g in frame.dropna(subset=["lap_time_s", "race_lap"]).groupby("stint_key"):
        if len(g) >= min_laps and g["race_lap"].nunique() >= 2:
            x, y = g["race_lap"].to_numpy(float), g["lap_time_s"].to_numpy(float)
            slopes.append(np.polyfit(x, y, 1)[0])
    return np.asarray(slopes, dtype=float)


def _quartiles(values: np.ndarray) -> dict[str, float | None]:
    if not values.size:
        return {"median": None, "q25": None, "q75": None}
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75)}


def warmup_visibility(
    frame: pd.DataFrame, phi: float, min_laps: int = 3, first_lap: int = 2, last_lap: int = 8
) -> dict[str, Any]:
    """S4 check on real stints: how often the corrected lap time shows an early dip (PRD S4, T3 test 3).

    ``frame`` holds scored laps only and needs ``stint_key``, ``stint_lap``, ``race_lap``, ``total_laps``
    and ``lap_time_s``. ``stint_lap`` is the 1-based position of the lap in its stint, counted over all
    laps of the stint before exclusions, so the out-lap (or race lap 1) is stint lap 1. Optional column
    ``TyreLife`` is summarized at the first scored lap.

    Per stint with at least ``min_laps`` scored laps, with
    ``y = lap_time_s - phi * (total_laps - race_lap)``:

    - dip: the fastest scored lap is at stint lap ``first_lap``..``last_lap`` and the last scored lap is
      slower than it (same definition as the synthetic acceptance test).
    - gain: y of the first scored lap minus the minimum y, in s, for dip stints. It is 0 when the first
      scored lap is itself the fastest, which happens when stint lap 1 is excluded and the dip is at lap 2.
    """
    cols = ["stint_key", "stint_lap", "race_lap", "total_laps", "lap_time_s"]
    sub = frame.dropna(subset=cols).copy()
    sub["y"] = sub["lap_time_s"] - phi * (sub["total_laps"] - sub["race_lap"])
    sub = sub.sort_values(["stint_key", "stint_lap"])
    n_stints, gains, first_laps, first_tyre_life = 0, [], [], []
    for _, g in sub.groupby("stint_key"):
        if len(g) < min_laps:
            continue
        n_stints += 1
        y = g["y"].to_numpy(dtype=float)
        laps = g["stint_lap"].to_numpy(dtype=float)
        first_laps.append(int(laps[0]))
        if "TyreLife" in g.columns and np.isfinite(float(g["TyreLife"].iloc[0])):
            first_tyre_life.append(float(g["TyreLife"].iloc[0]))
        i_min = int(np.argmin(y))
        if first_lap <= laps[i_min] <= last_lap and y[-1] > y[i_min]:
            gains.append(y[0] - y[i_min])
    gains_arr = np.asarray(gains, dtype=float)
    first_arr = np.asarray(first_laps, dtype=float)
    counts = pd.Series(first_laps, dtype=int).value_counts().sort_index()
    return {
        "n_stints": n_stints,
        "n_dip": int(gains_arr.size),
        "dip_share": gains_arr.size / n_stints if n_stints else None,
        "dip_gain_s": {"n": int(gains_arr.size), **_quartiles(gains_arr)},
        "first_scored_stint_lap": {
            "median": float(np.median(first_arr)) if first_arr.size else None,
            "counts": {str(k): int(v) for k, v in counts.items()},
        },
        "first_scored_tyre_life_median": float(np.median(first_tyre_life)) if first_tyre_life else None,
    }


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy and pandas scalars to JSON types. Non-finite floats become None."""
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    return value
