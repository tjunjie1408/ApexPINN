"""Fold-specific Gold features from verified Bronze through Silver (PLAN T10).

Normalization fits only eligible, scored development rows on declared training circuits. Transform
retains every Silver row. Quality failures have separate flags, never silently rewrite Silver filters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from apexpinn.contracts import Decisions, load_decisions
from apexpinn.data import manifest as mf
from apexpinn.data import probe
from apexpinn.data.silver import PRIMARY_KEY, SPEED_COLUMNS, build_silver

O1_MAX_TELEMETRY_MISSING = 0.20


def _decisions_hash(decisions: Decisions) -> str:
    return hashlib.sha256(json.dumps(asdict(decisions), sort_keys=True).encode()).hexdigest()


def _validate_silver(frame: pd.DataFrame, decisions: Decisions) -> None:
    if any(column.startswith("latent_") for column in frame.columns):
        raise ValueError("latent_* model outputs must never be stored in Silver or Gold")
    required = {
        *PRIMARY_KEY,
        "season",
        "circuit_id",
        "dataset_split",
        "split_fold",
        "event_is_dry",
        "is_scored_lap",
        "exclusion_reason",
        "proxy_telemetry_missing",
        "phi",
    }
    if missing := required - set(frame):
        raise ValueError(f"Silver missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Silver must not be empty")
    if frame[list(PRIMARY_KEY)].isna().any().any() or frame.duplicated(list(PRIMARY_KEY)).any():
        raise ValueError("missing or duplicate Silver primary keys")
    if frame.phi.nunique(dropna=False) != 1 or frame.phi.iloc[0] not in (
        decisions.phi_low,
        decisions.phi_mid,
        decisions.phi_high,
    ):
        raise ValueError("Gold requires one declared phi per dataset")
    for event, group in frame.groupby("session_id", sort=True):
        if event not in decisions.dev_events + decisions.holdout_events:
            raise ValueError(f"unconfigured event {event!r}")
        season, circuit = event.split("_", 1)
        expected_split = "development" if event in decisions.dev_events else "holdout"
        expected = {
            "season": int(season),
            "circuit_id": circuit,
            "split_fold": decisions.circuit_to_fold.get(circuit),
            "dataset_split": expected_split,
        }
        for column, value in expected.items():
            if value is None or not group[column].eq(value).all():
                raise ValueError(f"event {event}: inconsistent {column}")
    if frame.is_scored_lap.isna().any() or not frame.is_scored_lap.isin([True, False]).all():
        raise ValueError("is_scored_lap must contain nonmissing boolean flags")


def event_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    """O1 event gate over every original lap. Unknown telemetry status counts as missing."""
    rows = []
    for event, group in frame.groupby("session_id", sort=True):
        missing = ~group.proxy_telemetry_missing.eq(False).fillna(False)
        share = float(missing.mean())
        dry = bool(group.event_is_dry.eq(True).fillna(False).all())
        reason = "" if dry else "event_not_verified_dry"
        if not reason and share > O1_MAX_TELEMETRY_MISSING:
            reason = "event_telemetry_missing_gt_20pct"
        rows.append(
            {
                "session_id": event,
                "telemetry_missing_share": share,
                "is_dry": dry,
                "eligible": not reason,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _proxy_values(frame: pd.DataFrame, proxy: str) -> pd.Series:
    if proxy == "speed_trap":
        return probe.speed_trap_proxy(frame.rename(columns={v: k for k, v in SPEED_COLUMNS.items()}))
    column = f"proxy_{proxy}"
    if column not in frame:
        raise ValueError(f"missing {column}")
    return frame[column].astype(float)


def _valid_proxy(values: pd.Series) -> pd.Series:
    return np.isfinite(values) & values.ge(0)


def _normalize(values: pd.Series, lower: float, upper: float) -> pd.Series:
    normalized = (values - lower) / (upper - lower) if upper > lower else values * 0.0
    return normalized.clip(0.0, 1.0).where(_valid_proxy(values))


@dataclass(frozen=True)
class FoldNormalization:
    """Fitted training-only transform and provenance; no observed test-circuit context means."""

    test_fold: str
    proxy: str
    phi: float
    lower: float
    upper: float
    n_train: int
    training_events: tuple[str, ...]
    training_circuits: tuple[str, ...]
    circuit_q_means: tuple[tuple[str, float], ...]
    training_q_mean: float
    decisions_sha256: str
    training_rows_sha256: str

    @property
    def constant_range(self) -> bool:
        return self.upper == self.lower

    def context_q_mean(self, circuit: str) -> float:
        """Training circuit mean, or explicitly global training mean for an unseen circuit."""
        return dict(self.circuit_q_means).get(circuit, self.training_q_mean)


def fit_normalization(
    silver: pd.DataFrame,
    decisions: Decisions,
    *,
    test_fold: str,
    training_circuits: Sequence[str] | None = None,
) -> FoldNormalization:
    """Fit min/max on usable scored training rows. An explicit subset supports inner splits."""
    _validate_silver(silver, decisions)
    if test_fold not in decisions.folds:
        raise ValueError(f"unknown test_fold {test_fold!r}")
    allowed = set(decisions.circuit_to_fold) - set(decisions.folds[test_fold])
    selected = allowed if training_circuits is None else set(training_circuits)
    if not selected or not selected <= allowed:
        raise ValueError("training_circuits must be configured circuits excluding outer test circuits")
    proxy = str(decisions.heat_load_proxy.get("formula", ""))
    if (
        decisions.heat_load_proxy.get("normalization") != "per_fold_minmax"
        or proxy not in probe.PROXY_COLUMNS
    ):
        raise ValueError("Gold requires a declared proxy and per_fold_minmax normalization")
    train = silver.loc[silver.dataset_split.eq("development") & silver.circuit_id.isin(selected)].copy()
    if train.empty:
        raise ValueError(f"{test_fold}: no usable training rows")
    eligibility = event_eligibility(train).set_index("session_id")
    eligible = train.session_id.map(eligibility.eligible).eq(True)
    values = _proxy_values(train, proxy)
    use = (
        eligible
        & train.is_scored_lap.eq(True)
        & train.proxy_telemetry_missing.eq(False).fillna(False)
        & _valid_proxy(values)
    )
    train = train.loc[use].copy()
    if train.empty:
        raise ValueError(f"{test_fold}: no usable training rows")
    train["fitted_proxy"] = values.loc[use]
    train = train.sort_values(list(PRIMARY_KEY)).reset_index(drop=True)
    lower, upper = float(train.fitted_proxy.min()), float(train.fitted_proxy.max())
    q = _normalize(train.fitted_proxy, lower, upper)
    means = q.groupby(train.circuit_id).mean()
    fingerprint = pd.util.hash_pandas_object(train[[*PRIMARY_KEY, "fitted_proxy"]], index=False)
    return FoldNormalization(
        test_fold=test_fold,
        proxy=proxy,
        phi=float(train.phi.iloc[0]),
        lower=lower,
        upper=upper,
        n_train=len(train),
        training_events=tuple(sorted(train.session_id.unique())),
        training_circuits=tuple(sorted(train.circuit_id.unique())),
        circuit_q_means=tuple((str(c), float(v)) for c, v in means.sort_index().items()),
        training_q_mean=float(q.mean()),
        decisions_sha256=_decisions_hash(decisions),
        training_rows_sha256=hashlib.sha256(fingerprint.to_numpy().tobytes()).hexdigest(),
    )


def build_gold(silver: pd.DataFrame, stats: FoldNormalization, decisions: Decisions) -> pd.DataFrame:
    """Add bounded heat load, candidate inputs and visible feature-quality flags without dropping rows."""
    _validate_silver(silver, decisions)
    if stats.decisions_sha256 != _decisions_hash(decisions) or stats.phi != float(silver.phi.iloc[0]):
        raise ValueError("normalization decisions/phi do not match this dataset")
    frame = silver.copy(deep=True)
    values = _proxy_values(frame, stats.proxy)
    eligibility = event_eligibility(frame).set_index("session_id")
    reasons = frame.session_id.map(eligibility.reason)
    missing = ~frame.proxy_telemetry_missing.eq(False).fillna(False)
    reasons = reasons.mask(reasons.eq("") & missing, "missing_telemetry")
    reasons = reasons.mask(reasons.eq("") & ~_valid_proxy(values), "invalid_proxy")
    frame["proxy_speed_trap"] = _proxy_values(frame, "speed_trap")
    frame["proxy_q_heat"] = _normalize(values, stats.lower, stats.upper).where(reasons.eq(""))
    frame["proxy_q_clipped"] = _valid_proxy(values) & ((values < stats.lower) | (values > stats.upper))
    frame["feature_exclusion_reason"] = reasons
    frame["is_model_input_lap"] = frame.is_scored_lap.eq(True) & reasons.eq("")
    frame["normalization_test_fold"] = stats.test_fold
    frame["ctx_circuit_q_mean"] = frame.circuit_id.map(stats.context_q_mean)
    frame["ctx_q_mean_source"] = np.where(
        frame.circuit_id.isin(stats.training_circuits), "training_circuit", "global_training_fallback"
    )
    return frame


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fold_summary(gold: pd.DataFrame, stats: FoldNormalization) -> dict[str, Any]:
    usable = gold[gold.is_model_input_lap]
    training = usable[usable.session_id.isin(stats.training_events)]
    variance = []
    for (circuit, role), group in usable.groupby(["circuit_id", "fold_role"], sort=True):
        variance.append(
            {
                "circuit": circuit,
                "role": role,
                "laps": len(group),
                "q_mean": float(group.proxy_q_heat.mean()),
                "q_variance_ddof0": float(group.proxy_q_heat.var(ddof=0)),
                "clipped_laps": int(group.proxy_q_clipped.sum()),
            }
        )
    circuit_means = training.groupby("circuit_id").proxy_q_heat.mean()
    within = training.groupby("stint_id").proxy_q_heat.std(ddof=0).mean()
    across = circuit_means.std(ddof=1) if len(circuit_means) >= 2 else np.nan
    ratio = float(within / across) if np.isfinite(across) and across > 0 else None
    comparison_input = pd.DataFrame(
        {
            "event_id": training.session_id,
            "stint_key": training.stint_id,
            "race_lap": training.race_lap,
            "total_laps": training.race_length,
            "lap_time_s": training.obs_lap_time_s,
            **{
                name: _proxy_values(training, name).replace([np.inf, -np.inf], np.nan)
                for name in probe.PROXY_COLUMNS
            },
        }
    )
    comparisons = probe.compare_proxies(comparison_input, phi=stats.phi)
    selected = comparisons.get(stats.proxy, {})
    return _json_finite(
        {
            "normalization": {**asdict(stats), "constant_range": stats.constant_range},
            "rows": len(gold),
            "stints": int(gold.stint_id.nunique()),
            "scored_laps": int(gold.is_scored_lap.sum()),
            "model_input_laps": len(usable),
            "fold_population": [
                {
                    "split_fold": fold,
                    "rows": len(group),
                    "stints": int(group.stint_id.nunique()),
                    "scored_laps": int(group.is_scored_lap.sum()),
                    "model_input_laps": int(group.is_model_input_lap.sum()),
                }
                for fold, group in gold.groupby("split_fold", sort=True)
            ],
            "exclusion_counts": gold.exclusion_reason.value_counts().sort_index().to_dict(),
            "feature_exclusion_counts": gold.feature_exclusion_reason.value_counts().sort_index().to_dict(),
            "circuit_q_variance": variance,
            "l2_training_only": {
                "within_stint_std_mean": float(within),
                "across_circuit_mean_std_ddof1": across,
                "ratio": ratio,
                "limit_refuted": None if ratio is None else ratio > 0.1,
                "formula": "mean(stint q std ddof=0) / std(training circuit q means, ddof=1)",
            },
            "training_proxy_checks": comparisons,
            "o2_status": "REVIEW_REQUIRED" if selected.get("leak_warning", False) else "PARTIAL_VALIDATION",
        }
    )


def build_dataset(
    *,
    bronze_root: Path,
    out_root: Path,
    config: Path,
    allocations: Path,
    event_ids: Sequence[str],
    folds: Sequence[str],
    phi_setting: str,
    report_path: Path,
) -> dict[str, Any]:
    """Rebuild verified Bronze -> Silver -> fold Gold; a failed fold never fits on its own test rows.

    Partial development coverage is supported and recorded, not promoted to G0 completion. The CLI
    returns failure when any requested fold fails. The root manifest is authoritative for this build.
    """
    decisions = load_decisions(config)
    if phi_setting not in ("low", "mid", "high"):
        raise ValueError("phi_setting must be low, mid or high")
    if (
        not event_ids
        or len(set(event_ids)) != len(event_ids)
        or not set(event_ids) <= set(decisions.dev_events)
    ):
        raise ValueError(
            "event_ids must be unique configured development events; holdout processing disabled"
        )
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(decisions.folds):
        raise ValueError("folds must be unique configured test folds")
    directory = out_root / f"phi_{phi_setting}"
    if directory.resolve().is_relative_to(bronze_root.resolve()) or report_path.resolve().is_relative_to(
        bronze_root.resolve()
    ):
        raise ValueError("Gold output/report must not overwrite Bronze")
    allocation_table = yaml.safe_load(allocations.read_text(encoding="utf-8"))["events"]
    required = ("laps.parquet", "weather.parquet", "lap_telemetry.parquet")
    inputs, frames = {}, []
    for event in sorted(event_ids):
        event_dir = bronze_root / event
        if problems := mf.verify_event(event_dir, required_files=required):
            raise ValueError(f"{event}: Bronze integrity failed: {problems}")
        manifest = mf.read_manifest(event_dir)
        if manifest is None or manifest.get("event_id") != event:
            raise ValueError(f"{event}: Bronze event_id mismatch")
        source = mf.sha256_file(event_dir / "manifest.json")
        inputs[event] = {"bronze_directory": str(event_dir), "manifest_sha256": source, "manifest": manifest}
        frames.append(
            build_silver(
                *(pd.read_parquet(event_dir / name) for name in required),
                event_id=event,
                decisions=decisions,
                allocation=allocation_table[event],
                phi=getattr(decisions, f"phi_{phi_setting}"),
                source_version=source,
            )
        )
    silver = pd.concat(frames, ignore_index=True).sort_values(list(PRIMARY_KEY)).reset_index(drop=True)
    _validate_silver(silver, decisions)
    results, failures = {}, {}
    (directory / "manifest.json").unlink(missing_ok=True)
    for fold in sorted(folds):
        try:
            stats = fit_normalization(silver, decisions, test_fold=fold)
        except ValueError as exc:
            failures[fold] = str(exc)
            continue
        gold = build_gold(silver, stats, decisions)
        gold["fold_role"] = np.where(gold.split_fold.eq(fold), "test", "train")
        summary = _fold_summary(gold, stats)
        fold_dir = directory / fold
        (fold_dir / "manifest.json").unlink(missing_ok=True)
        metadata = mf.write_parquet(gold, fold_dir / "laps.parquet")
        summary["files"] = {"laps.parquet": metadata}
        mf.write_manifest(fold_dir, {"schema": "apexpinn.gold_fold/v1", "test_fold": fold, **summary})
        results[fold] = summary
    coverage = {
        "configured_dev_events_complete": set(event_ids) == set(decisions.dev_events),
        "configured_folds_complete": set(results) == set(decisions.folds),
        "requested_events": sorted(event_ids),
        "requested_folds": sorted(folds),
        "missing_dev_events": sorted(set(decisions.dev_events) - set(event_ids)),
    }
    dataset_hash = hashlib.sha256(
        json.dumps(
            {fold: info["files"]["laps.parquet"]["sha256"] for fold, info in sorted(results.items())},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    from apexpinn.eval.data_report import build_data_report

    sources = (
        Path(__file__),
        Path(probe.__file__),
        Path(__file__).with_name("silver.py"),
        Path(mf.__file__),
        Path(__file__).parents[1] / "eval" / "data_report.py",
    )
    receipt = {
        "schema": "apexpinn.gold_build/v1",
        "evidence": "Observed",
        "quantity": "Gold features and G0 evidence",
        "population": "configured development events supplied explicitly; no temporal holdout",
        "conditions": {
            "phi": getattr(decisions, f"phi_{phi_setting}"),
            "proxy": decisions.heat_load_proxy["formula"],
            "normalization": "eligible scored training circuits only; clipped minmax [0,1]",
            "context_fallback": "global training mean for unseen circuits, not an observed circuit mean",
            "constant_range": "finite values map to zero; range degeneracy is visible in statistics",
            "forecasting_evaluation": "NOT RUN",
        },
        "instrument": {
            "pandas": pd.__version__,
            "source_sha256": {p.name: mf.sha256_file(p) for p in sources},
            "config_sha256": mf.sha256_file(config),
            "allocations_sha256": mf.sha256_file(allocations),
        },
        "inputs": inputs,
        "coverage": coverage,
        "event_eligibility": event_eligibility(silver).to_dict("records"),
        "folds": results,
        "fold_failures": failures,
        "dataset_sha256": dataset_hash,
        "g0_status": "NOT ESTABLISHED: requires coverage, independent rebuild hash checks and T9/T10 tests",
    }
    mf.write_manifest(directory, receipt)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        build_data_report(receipt, source=str(directory / "manifest.json")), encoding="utf-8"
    )
    temporary.replace(report_path)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-root", type=Path, default=Path("data/bronze"))
    parser.add_argument("--out-root", type=Path, default=Path("data/gold"))
    parser.add_argument("--config", type=Path, default=Path("configs/decisions_v1.yaml"))
    parser.add_argument("--allocations", type=Path, default=Path("configs/compound_allocation.yaml"))
    parser.add_argument("--event", action="append", required=True)
    parser.add_argument("--fold", action="append")
    parser.add_argument("--phi", choices=("low", "mid", "high"), default="mid")
    parser.add_argument("--report", type=Path, default=Path("reports/data_report.md"))
    args = parser.parse_args(argv)
    receipt = build_dataset(
        bronze_root=args.bronze_root,
        out_root=args.out_root,
        config=args.config,
        allocations=args.allocations,
        event_ids=args.event,
        folds=args.fold or tuple(load_decisions(args.config).folds),
        phi_setting=args.phi,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "dataset_sha256": receipt["dataset_sha256"],
                "folds": sorted(receipt["folds"]),
                "fold_failures": receipt["fold_failures"],
                "g0_status": receipt["g0_status"],
            },
            sort_keys=True,
        )
    )
    return int(bool(receipt["fold_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
