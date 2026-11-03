"""Deterministic Bronze-to-Silver preparation. No normalization or model fitting (PLAN T9).

All original laps remain present, including rejected laps. ``proxy_y`` is the corrected target in
seconds. Raw heat-load aggregates are not normalized here; T10 supplies ``proxy_q_heat`` to guards.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from apexpinn.contracts import Decisions, load_decisions
from apexpinn.data import manifest as mf
from apexpinn.data.probe import WET_COMPOUNDS, exclusion_reason, gap_to_car_ahead

PRIMARY_KEY = ("session_id", "driver_id", "stint_id", "lap_number")
LAP_COLUMNS = {
    "LapTime_s": "obs_lap_time_s",
    "LapStartTime_s": "obs_lap_start_s",
    "Time_s": "obs_lap_end_s",
    "PitInTime_s": "obs_pit_in_s",
    "PitOutTime_s": "obs_pit_out_s",
    "TrackStatus": "obs_track_status",
    "IsAccurate": "obs_is_accurate",
    "Deleted": "obs_deleted",
    "Compound": "obs_compound",
    "TyreLife": "obs_tyre_life",
    "FreshTyre": "obs_fresh_tyre",
}
TELEMETRY_COLUMNS = (
    "energy_index",
    "brake_fraction",
    "brake_fraction_weighted",
    "telemetry_missing",
)
OPTIONAL_TELEMETRY_COLUMNS = ("n_car", "n_pos", "car_coverage", "pos_coverage", "speed_mean_kmh")
SPEED_COLUMNS = {
    "SpeedI1": "obs_speed_i1_kmh",
    "SpeedI2": "obs_speed_i2_kmh",
    "SpeedFL": "obs_speed_fl_kmh",
    "SpeedST": "obs_speed_st_kmh",
}


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _positive_ints(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric).all() or (numeric < 1).any() or (numeric % 1 != 0).any():
        raise ValueError(f"{name} must contain finite positive integers")
    return numeric.astype(np.int64)


def _lap_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(frame, ("DriverNumber", "LapNumber"), name)
    result = frame.copy().reset_index(drop=True)
    if result["DriverNumber"].isna().any() or result["DriverNumber"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} has missing driver identifiers")
    result["DriverNumber"] = result["DriverNumber"].astype(str)
    result["LapNumber"] = _positive_ints(result["LapNumber"], f"{name}.LapNumber")
    if result.duplicated(["DriverNumber", "LapNumber"]).any():
        raise ValueError(f"{name} has duplicate driver/lap primary keys")
    return result


def _track_temperature(laps: pd.DataFrame, weather: pd.DataFrame) -> np.ndarray:
    """Most recent recorded weather at or before lap start, never a later sample."""
    if weather.empty:
        return np.full(len(laps), np.nan)
    _require_columns(weather, ("Time_s", "TrackTemp"), "weather")
    samples = weather.dropna(subset=["Time_s"]).sort_values("Time_s", kind="stable")
    samples = samples.drop_duplicates("Time_s", keep="last")
    times = samples["Time_s"].to_numpy(dtype=float)
    temperatures = samples["TrackTemp"].to_numpy(dtype=float)
    starts = laps["LapStartTime_s"].to_numpy(dtype=float)
    indices = np.searchsorted(times, starts, side="right") - 1
    result = np.full(len(laps), np.nan)
    known = (indices >= 0) & np.isfinite(starts)
    result[known] = temperatures[indices[known]]
    return result


def build_silver(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    telemetry: pd.DataFrame,
    *,
    event_id: str,
    decisions: Decisions,
    allocation: Mapping[str, str],
    phi: float,
    source_version: str,
) -> pd.DataFrame:
    """Prepare one configured race, retaining rejected rows and original race/stint lap counters.

    ``source_version`` identifies the verified Bronze manifest. Circuit folds are shared across seasons;
    ``dataset_split`` separately marks development/holdout. Preparing holdout metadata is not evaluation.
    """
    if event_id not in decisions.dev_events + decisions.holdout_events:
        raise ValueError(f"event {event_id!r} is not configured")
    if phi not in (decisions.phi_low, decisions.phi_mid, decisions.phi_high):
        raise ValueError("phi must be one of the declared sensitivity values")
    if not source_version:
        raise ValueError("source_version must identify the Bronze source")
    compounds = [allocation.get(name, "") for name in ("HARD", "MEDIUM", "SOFT")]
    if any(c not in {f"C{i}" for i in range(1, 7)} for c in compounds):
        raise ValueError("allocation must supply HARD/MEDIUM/SOFT C-numbers")
    if not compounds[0] < compounds[1] < compounds[2]:
        raise ValueError("allocation requires distinct HARD < MEDIUM < SOFT C-numbers")
    raw = _lap_keys(laps, "laps")
    if raw.empty:
        raise ValueError("laps must not be empty")
    _require_columns(raw, ("Stint", *LAP_COLUMNS), "laps")
    raw["Stint"] = _positive_ints(raw["Stint"], "Stint")
    raw = raw.sort_values(["DriverNumber", "LapNumber"], kind="stable").reset_index(drop=True)
    if raw.groupby("DriverNumber")["Stint"].diff().lt(0).any():
        raise ValueError("Stint identifiers must be monotonic within each driver")
    telem = _lap_keys(telemetry, "telemetry")
    _require_columns(telem, TELEMETRY_COLUMNS, "telemetry")
    keys = ["DriverNumber", "LapNumber"]
    if set(map(tuple, raw[keys].to_numpy())) != set(map(tuple, telem[keys].to_numpy())):
        raise ValueError("telemetry driver/lap keys must exactly match laps")
    telemetry_columns = (*TELEMETRY_COLUMNS, *(c for c in OPTIONAL_TELEMETRY_COLUMNS if c in telem))
    joined = raw[keys].merge(telem[[*keys, *telemetry_columns]], on=keys, validate="one_to_one")

    reason_inputs = raw.rename(
        columns={
            "LapTime_s": "lap_time_s",
            "Time_s": "lap_end_s",
            "PitInTime_s": "pit_in_s",
            "PitOutTime_s": "pit_out_s",
        }
    ).copy()
    reason_inputs["gap_ahead_s"] = gap_to_car_ahead(reason_inputs)
    reasons = exclusion_reason(reason_inputs, decisions.traffic_gap_threshold_s)
    observed_compound = raw["Compound"].astype("string").str.upper()
    compound_c = observed_compound.map(allocation)
    known_dry = observed_compound.isin(("HARD", "MEDIUM", "SOFT"))
    event_is_dry = bool(known_dry.all())
    invalid_time = ~np.isfinite(raw["LapTime_s"]) | raw["LapTime_s"].le(0)
    reasons = reasons.mask(reasons.eq("") & invalid_time, "invalid_lap_time")
    reasons = reasons.mask(reasons.eq("") & compound_c.isna(), "unknown_compound")
    # A wet race remains inspectable in Silver, but never contributes dry-race scoring rows.
    if not event_is_dry:
        reasons = reasons.mask(
            reasons.eq(""),
            "wet_event" if observed_compound.isin(WET_COMPOUNDS).any() else "unknown_event_compounds",
        )

    season_text, circuit = event_id.split("_", 1)
    if circuit not in decisions.circuit_to_fold:
        raise ValueError(f"circuit {circuit!r} has no fold")
    frame = pd.DataFrame(
        {
            "session_id": event_id,
            "season": int(season_text),
            "circuit_id": circuit,
            "driver_id": raw["DriverNumber"],
            "stint_id": event_id + ":" + raw["DriverNumber"] + ":" + raw["Stint"].astype(str),
            "lap_number": raw["LapNumber"],
            "race_lap": raw["LapNumber"],
            "race_length": int(raw["LapNumber"].max()),
            "split_fold": decisions.circuit_to_fold[circuit],
            "dataset_split": "holdout" if event_id in decisions.holdout_events else "development",
            "event_is_dry": event_is_dry,
            "compound_c": compound_c,
            "source_version": source_version,
            "phi": float(phi),
            "exclusion_reason": reasons,
            "is_scored_lap": reasons.eq(""),
        }
    )
    frame["stint_lap"] = frame["race_lap"] - frame.groupby("stint_id")["race_lap"].transform("min") + 1
    for raw_name, silver_name in LAP_COLUMNS.items():
        frame[silver_name] = raw[raw_name]
    for raw_name, silver_name in SPEED_COLUMNS.items():
        frame[silver_name] = raw[raw_name] if raw_name in raw else np.nan
    frame["obs_gap_ahead_s"] = reason_inputs["gap_ahead_s"]
    frame["obs_track_temp_c"] = _track_temperature(raw, weather)
    for name in telemetry_columns:
        frame[f"proxy_{name}"] = joined[name]
    frame["proxy_y"] = frame["obs_lap_time_s"] - phi * (frame["race_length"] - frame["race_lap"])
    if frame.duplicated(list(PRIMARY_KEY)).any():
        raise ValueError("duplicate Silver primary keys")
    return frame.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Build one verified Bronze event; repeat writes are deterministic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--bronze-root", type=Path, default=Path("data/bronze"))
    parser.add_argument("--out-root", type=Path, default=Path("data/silver"))
    parser.add_argument("--config", type=Path, default=Path("configs/decisions_v1.yaml"))
    parser.add_argument("--allocations", type=Path, default=Path("configs/compound_allocation.yaml"))
    parser.add_argument("--phi", choices=("low", "mid", "high"), default="mid")
    args = parser.parse_args(argv)
    bronze_dir = args.bronze_root / args.event
    output_dir = args.out_root / args.event / f"phi_{args.phi}"
    required = ("laps.parquet", "weather.parquet", "lap_telemetry.parquet")
    problems = mf.verify_event(bronze_dir, required_files=required)
    if problems:
        raise ValueError(f"Bronze integrity failed: {problems}")
    manifest = mf.read_manifest(bronze_dir)
    if manifest is None or manifest["event_id"] != args.event:
        raise ValueError("Bronze manifest event_id does not match the requested event")
    if output_dir.resolve().is_relative_to(bronze_dir.resolve()):
        raise ValueError("Silver output must not overwrite Bronze")
    decisions = load_decisions(args.config)
    allocations = yaml.safe_load(args.allocations.read_text(encoding="utf-8"))["events"]
    source_version = mf.sha256_file(bronze_dir / "manifest.json")
    frame = build_silver(
        *(pd.read_parquet(bronze_dir / name) for name in required),
        event_id=args.event,
        decisions=decisions,
        allocation=allocations[args.event],
        phi=getattr(decisions, f"phi_{args.phi}"),
        source_version=source_version,
    )
    (output_dir / "manifest.json").unlink(missing_ok=True)
    metadata = mf.write_parquet(frame, output_dir / "laps.parquet")
    receipt = {
        "schema": "apexpinn.silver_manifest/v1",
        "event_id": args.event,
        "phi": float(frame.phi.iloc[0]),
        "bronze_manifest_sha256": source_version,
        "config_sha256": mf.sha256_file(args.config),
        "allocations_sha256": mf.sha256_file(args.allocations),
        "silver_source_sha256": mf.sha256_file(Path(__file__)),
        "files": {"laps.parquet": metadata},
        "scored_laps": int(frame.is_scored_lap.sum()),
        "exclusion_counts": frame.exclusion_reason.value_counts().sort_index().to_dict(),
        "normalization": "NOT FITTED; proxy_q_heat is supplied by T10",
    }
    mf.write_manifest(output_dir, receipt)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "rows": len(frame),
                "scored_laps": receipt["scored_laps"],
                "sha256": metadata["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
