"""Rolling original-lap origins, four-field evidence and strict paired populations (PLAN T11).

The caller supplies already-fitted fold-specific models. This module never calls fit or tunes a model.
Draft configurations are allowed only for explicitly labeled validation, not empirical acceptance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from apexpinn.contracts import Decisions, MetricRecord, PredictionOutput
from apexpinn.data.gold import FoldNormalization
from apexpinn.data.guards import audit_origins, slice_at_origin
from apexpinn.data.silver import PRIMARY_KEY
from apexpinn.eval import metrics
from apexpinn.eval.bootstrap import BootstrapResult, paired_bootstrap_rel_rmse
from apexpinn.models.stints import Forecaster, StintHistory
from apexpinn.models.stints import anchor_offset as _array_anchor


@dataclass(frozen=True)
class ProtocolConfig:
    decisions: Decisions
    model_name: str
    phi: float
    validation_only: bool = False


@dataclass(frozen=True)
class EvaluationFold:
    """One fold's prepared Gold and declared provenance. Caller owns fitted-model provenance."""

    gold: pd.DataFrame
    normalization: FoldNormalization
    gold_hash: str
    model_hash: str
    inner_validation_fold: str | None = None


@dataclass(frozen=True)
class InnerSplit:
    outer_fold: str
    validation_fold: str
    training_circuits: tuple[str, ...]
    validation_circuits: tuple[str, ...]
    quantity: dict[str, Any]
    population: dict[str, Any]
    conditions: dict[str, Any]
    instrument: dict[str, Any]
    status: str = "Assumed"


def inner_circuit_split(decisions: Decisions, outer_fold: str) -> InnerSplit:
    """One fixed inner holdout group: next group in declared config order, excluding the outer test.

    Refit normalization with training_circuits on every inner fit. This declares the split, not the O7
    tuner. A single holdout bounds cost but has more tuning variance than full inner cross-validation.
    """
    order = tuple(decisions.folds)
    if outer_fold not in order or len(order) < 3:
        raise ValueError("inner split requires a configured outer fold and at least three circuit groups")
    validation_fold = order[(order.index(outer_fold) + 1) % len(order)]
    training = tuple(
        sorted(
            c for fold in order if fold not in (outer_fold, validation_fold) for c in decisions.folds[fold]
        )
    )
    validation = tuple(decisions.folds[validation_fold])
    return InnerSplit(
        outer_fold,
        validation_fold,
        training,
        validation,
        quantity={"protocol": "hyperparameter inner circuit split"},
        population={
            "development_seasons": list(decisions.dev_seasons),
            "outer_test_circuits": list(decisions.folds[outer_fold]),
        },
        conditions={
            "method": "cyclic next configured group as inner holdout/v1",
            "same_split_across_models_and_phi": True,
            "normalization_refit_on_inner_training_only": True,
        },
        instrument={"config_fold_order": list(order), "tuner_execution": "NOT IMPLEMENTED"},
    )


@dataclass(frozen=True)
class ForecastRecord:
    """One exact-horizon target. Intermediate targets are never pooled into this horizon's score."""

    model: str
    fold: str
    event_id: str
    stint_id: str
    t0: int
    horizon: int
    target_stint_lap: int
    phi: float
    y_true: float
    y_pred: float
    sigma_s: float
    member_means_s: tuple[float, ...]
    member_sigma_s: tuple[float, ...] | None
    quantity: dict[str, Any]
    population: dict[str, Any]
    conditions: dict[str, Any]
    instrument: dict[str, Any]
    status: str = "Observed"

    @property
    def key(self) -> tuple[str, str, str, int, int, float]:
        return self.fold, self.event_id, self.stint_id, self.t0, self.horizon, self.phi


@dataclass(frozen=True)
class OriginAudit:
    fold: str
    event_id: str
    stint_id: str
    t0: int
    horizon: int
    eligible: bool
    reason: str
    quantity: dict[str, Any]
    population: dict[str, Any]
    conditions: dict[str, Any]
    instrument: dict[str, Any]
    status: str = "Observed"


def anchor_offset(history: StintHistory, shape_fn: Callable[[np.ndarray], Any]) -> tuple[float, float]:
    """Model-independent adapter to the canonical shared array anchor; history is the only target input."""
    shape = np.asarray(shape_fn(history.tau), dtype=np.float64)
    if not np.isfinite(shape).all():
        raise ValueError("historical shape must be finite")
    return _array_anchor(history.y, shape)


def _validate_fold(fold: str, data: EvaluationFold, cfg: ProtocolConfig) -> pd.DataFrame:
    frame, stats, decisions = data.gold, data.normalization, cfg.decisions
    required = {
        *PRIMARY_KEY,
        "dataset_split",
        "split_fold",
        "circuit_id",
        "phi",
        "ctx_circuit_q_mean",
        "normalization_test_fold",
        "ctx_q_mean_source",
    }
    if missing := required - set(frame):
        raise ValueError(f"Gold missing columns: {sorted(missing)}")
    if (
        frame.empty
        or frame.duplicated(list(PRIMARY_KEY)).any()
        or frame[list(PRIMARY_KEY)].isna().any().any()
    ):
        raise ValueError("Gold is empty or has missing/duplicate primary keys")
    if not data.gold_hash or not data.model_hash:
        raise ValueError("Gold/model source hashes must be declared")
    if fold not in decisions.folds or stats.test_fold != fold:
        raise ValueError("fold does not match its normalization")
    if not frame.normalization_test_fold.eq(fold).all():
        raise ValueError("normalization_test_fold does not match the requested fold")
    expected_config = hashlib.sha256(json.dumps(asdict(decisions), sort_keys=True).encode()).hexdigest()
    if stats.decisions_sha256 != expected_config or stats.phi != cfg.phi or not frame.phi.eq(cfg.phi).all():
        raise ValueError("Gold normalization decisions/phi do not match the protocol")
    if set(stats.training_circuits) & set(decisions.folds[fold]) or not set(stats.training_events) <= set(
        decisions.dev_events
    ):
        raise ValueError("normalization training provenance includes test circuits or holdout events")
    event_circuits = {event.split("_", 1)[1] for event in stats.training_events}
    if not event_circuits or event_circuits != set(stats.training_circuits):
        raise ValueError("normalization training provenance has inconsistent events/circuits")
    if not frame.dataset_split.eq("development").all() or not set(frame.session_id) <= set(
        decisions.dev_events
    ):
        raise ValueError("temporal holdout evaluation is disabled in this development runner")
    for event, group in frame.groupby("session_id", sort=True):
        _, circuit = event.split("_", 1)
        if (
            not group.circuit_id.eq(circuit).all()
            or not group.split_fold.eq(decisions.circuit_to_fold[circuit]).all()
        ):
            raise ValueError("event/circuit/fold metadata is inconsistent")
    evaluated_fold = fold
    if data.inner_validation_fold is not None:
        split = inner_circuit_split(decisions, fold)
        if data.inner_validation_fold != split.validation_fold:
            raise ValueError("inner validation fold differs from the declared fixed split")
        if set(stats.training_circuits) & set(split.validation_circuits):
            raise ValueError("inner validation circuits entered the normalization fit")
        evaluated_fold = split.validation_fold
    test = frame.loc[frame.split_fold.eq(evaluated_fold)].copy()
    if not test.ctx_q_mean_source.eq("global_training_fallback").all():
        raise ValueError("unseen test circuits must use the declared global training context fallback")
    if not test.ctx_circuit_q_mean.eq(stats.training_q_mean).all():
        raise ValueError("test-circuit context mean is not the fitted training fallback")
    return test


def _fields(
    cfg: ProtocolConfig, fold: str, data: EvaluationFold, event: str, stint: str, t0: int, horizon: int
) -> dict[str, Any]:
    return {
        "quantity": {"target": "race-lap-trend-corrected lap time", "unit": "s"},
        "population": {
            "fold": fold,
            "event_id": event,
            "stint_id": stint,
            "t0_original_lap": t0,
            "horizon_actual_laps": horizon,
        },
        "conditions": {
            "model": cfg.model_name,
            "phi": cfg.phi,
            "k_min": cfg.decisions.k_min,
            "validation_only": cfg.validation_only,
            "config_status": cfg.decisions.status,
            "evaluation_stage": "inner_validation" if data.inner_validation_fold else "outer_test",
        },
        "instrument": {
            "method": "rolling original-lap protocol/v2",
            "gold_hash": data.gold_hash,
            "model_hash": data.model_hash,
            "normalization_hash": data.normalization.training_rows_sha256,
            "evaluated_fold": data.inner_validation_fold or fold,
        },
    }


def _component_scales(prediction: PredictionOutput, horizon: int) -> tuple[float, ...] | None:
    if prediction.member_sigma_s is not None:
        return tuple(float(v) for v in prediction.member_sigma_s[:, horizon - 1])
    if prediction.member_means_s.shape[0] == 1:
        if not np.allclose(prediction.mean_s, prediction.member_means_s[0], rtol=1e-6, atol=1e-8):
            raise ValueError("single-member mean disagrees with prediction mean_s")
        return (float(prediction.sigma_s[horizon - 1]),)
    return None


def run_protocol(
    forecaster: Forecaster | Mapping[str, Forecaster],
    folds: Mapping[str, EvaluationFold],
    cfg: ProtocolConfig,
    *,
    audit: list[OriginAudit],
    horizons: Sequence[int] | None = None,
) -> list[ForecastRecord]:
    """Evaluate prepared development test rows, appending every candidate's decision to explicit audit.

    Input-quality failures are separate from population rejections. Model failures propagate; a failed
    model never obtains a smaller favorable population by silently dropping its unsuccessful forecasts.
    """
    if (
        not cfg.model_name
        or not np.isfinite(cfg.phi)
        or cfg.phi
        not in (
            cfg.decisions.phi_low,
            cfg.decisions.phi_mid,
            cfg.decisions.phi_high,
        )
    ):
        raise ValueError("model_name and a declared phi are required")
    if cfg.decisions.status == "draft" and not cfg.validation_only:
        raise ValueError("draft config requires explicit validation_only=True; G0/freeze remain open")
    if not folds:
        raise ValueError("at least one prepared fold is required")
    evaluated = cfg.decisions.horizons if horizons is None else horizons
    if (
        not evaluated
        or len(set(evaluated)) != len(evaluated)
        or not set(evaluated) <= set(cfg.decisions.horizons)
    ):
        raise ValueError("evaluated horizons must be a unique declared subset")
    if isinstance(forecaster, Mapping):
        if set(forecaster) != set(folds):
            raise ValueError("forecasters must match the fold-specific prepared data")
        models = forecaster
    elif len(folds) == 1:
        models = {next(iter(folds)): forecaster}
    else:
        raise ValueError("multiple folds require fold-specific pre-fitted forecasters")
    prepared = {fold: _validate_fold(fold, data, cfg) for fold, data in sorted(folds.items())}
    records = []
    for fold, test in prepared.items():
        if test.empty:
            continue
        groups = {str(key): group for key, group in test.groupby("stint_id", sort=True)}
        candidates = audit_origins(test, horizons=evaluated, k_min=cfg.decisions.k_min)
        for origin in candidates.itertuples(index=False):
            fields = _fields(
                cfg, fold, folds[fold], origin.session_id, origin.stint_id, origin.t0, origin.horizon
            )
            reason = origin.reason
            if not reason:
                try:
                    stint = groups[origin.stint_id]
                    history, design, labels = slice_at_origin(
                        stint,
                        origin.t0,
                        origin.horizon,
                        k_min=cfg.decisions.k_min,
                        circuit_q_mean=folds[fold].normalization.training_q_mean,
                    )
                except ValueError as exc:
                    reason = f"input_quality: {exc}"
                else:
                    if not np.isfinite(labels[-1]):
                        reason = "target_nonfinite"
                    else:
                        output = models[fold].forecast(history, origin.horizon, design.scenario)
                        if not isinstance(output, PredictionOutput) or output.mean_s.shape != (
                            origin.horizon,
                        ):
                            raise ValueError("forecaster returned an invalid PredictionOutput/horizon")
                        scales = _component_scales(output, origin.horizon)
                        records.append(
                            ForecastRecord(
                                model=cfg.model_name,
                                fold=fold,
                                event_id=origin.session_id,
                                stint_id=origin.stint_id,
                                t0=origin.t0,
                                horizon=origin.horizon,
                                target_stint_lap=origin.t0 + origin.horizon,
                                phi=cfg.phi,
                                y_true=float(labels[-1]),
                                y_pred=float(output.mean_s[-1]),
                                sigma_s=float(output.sigma_s[-1]),
                                member_means_s=tuple(float(v) for v in output.member_means_s[:, -1]),
                                member_sigma_s=scales,
                                **fields,
                            )
                        )
            fields["quantity"] = {"metric": "origin_eligibility", "unit": "boolean"}
            audit.append(
                OriginAudit(
                    fold,
                    origin.session_id,
                    origin.stint_id,
                    origin.t0,
                    origin.horizon,
                    not reason,
                    reason,
                    **fields,
                )
            )
    return records


def _record_index(records: list[ForecastRecord]) -> dict[tuple, ForecastRecord]:
    index = {r.key: r for r in records}
    if len(index) != len(records):
        raise ValueError("duplicate forecast keys")
    return index


def summarize_origins(audit: list[OriginAudit]) -> list[MetricRecord]:
    """Counts and history-rejection share on the original candidate denominator, per model/fold/phi/h."""
    if not audit:
        raise ValueError("cannot summarize an empty origin audit")
    groups: dict[tuple, list[OriginAudit]] = {}
    seen = set()
    for row in audit:
        key = (row.conditions["model"], row.fold, row.conditions["phi"], row.horizon)
        identity = (*key, row.event_id, row.stint_id, row.t0)
        if identity in seen:
            raise ValueError("duplicate origin audit keys")
        seen.add(identity)
        groups.setdefault(key, []).append(row)
    results = []
    for (model, fold, phi, horizon), group in sorted(groups.items()):
        history_rejected = sum(r.reason == "history_not_consecutive_clean" for r in group)
        values = {
            "candidate_origins": len(group),
            "eligible_origins": sum(r.eligible for r in group),
            "history_rejected_origins": history_rejected,
            "input_quality_rejected_origins": sum(r.reason.startswith("input_quality:") for r in group),
            "target_rejected_origins": sum(r.reason.startswith("target_") for r in group),
            "history_rejection_share": history_rejected / len(group),
        }
        for name, value in values.items():
            results.append(
                MetricRecord(
                    quantity={"metric": name, "unit": "fraction" if name.endswith("share") else "count"},
                    population={
                        "fold": fold,
                        "events": sorted({r.event_id for r in group}),
                        "denominator": "all original t0 from k_min through max_stint_lap-h in each stint",
                    },
                    conditions={
                        "model": model,
                        "phi": phi,
                        "horizon_actual_laps": horizon,
                        "validation_only": group[0].conditions["validation_only"],
                    },
                    instrument={
                        "method": "original-lap origin audit/v1",
                        "gold_hashes": sorted({r.instrument["gold_hash"] for r in group}),
                    },
                    value=float(value),
                    status="Observed",
                )
            )
    return results


def score_forecasts(records: list[ForecastRecord]) -> list[MetricRecord]:
    """Pool exact endpoint records by model/phi/horizon; unknown mixtures do not get invented scales."""
    if not records:
        raise ValueError("cannot score an empty forecast population")
    groups: dict[tuple, list[ForecastRecord]] = {}
    for record in records:
        groups.setdefault((record.model, record.phi, record.horizon), []).append(record)
    results = []
    for (model, phi, horizon), group in sorted(groups.items()):
        index = _record_index(group)
        group = [index[key] for key in sorted(index)]
        y, predicted = [r.y_true for r in group], [r.y_pred for r in group]
        population = {
            "n_forecasts": len(group),
            "events": sorted({r.event_id for r in group}),
            "record_keys_sha256": hashlib.sha256(json.dumps(sorted(index)).encode()).hexdigest(),
        }
        if any(r.conditions != group[0].conditions for r in group):
            raise ValueError("cannot pool forecasts with different protocol conditions")
        conditions = {
            **group[0].conditions,
            "model": model,
            "phi": phi,
            "horizon_actual_laps": horizon,
            "validation_only": group[0].conditions["validation_only"],
        }
        instrument = {
            "method": "endpoint metrics/v1",
            "gold_hashes": sorted({r.instrument["gold_hash"] for r in group}),
            "model_hashes": sorted({r.instrument["model_hash"] for r in group}),
        }
        values = {"rmse": metrics.rmse(y, predicted), "mae": metrics.mae(y, predicted)}
        known = all(r.member_sigma_s is not None for r in group)
        if known:
            values["crps"] = float(
                np.mean(
                    [
                        metrics.crps_gaussian_mixture(r.y_true, r.member_means_s, r.member_sigma_s)
                        for r in group
                    ]
                )
            )
            for level in (0.5, 0.9):
                bounds = [
                    metrics.central_interval(r.member_means_s, r.member_sigma_s, level=level) for r in group
                ]
                lower, upper = np.asarray(bounds).T
                values[f"coverage_{int(level * 100)}"] = metrics.coverage(y, lower, upper)
                values[f"width_{int(level * 100)}"] = metrics.interval_width(lower, upper)
        else:
            instrument["probabilistic_status"] = "Unknown: ensemble member scales not supplied"
            values.update(
                {name: None for name in ("crps", "coverage_50", "coverage_90", "width_50", "width_90")}
            )
        for name, value in values.items():
            results.append(
                MetricRecord(
                    quantity={"metric": name, "unit": "fraction" if name.startswith("coverage") else "s"},
                    population=population,
                    conditions=conditions,
                    instrument=instrument,
                    value=value,
                    status="Unknown" if value is None else "Observed",
                )
            )
    return results


def paired_comparison(
    model: list[ForecastRecord],
    baseline: list[ForecastRecord],
    *,
    horizon: int,
    phi: float,
    seed: int,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Require identical keyed populations/targets/provenance, never intersect mismatched records."""
    left = _record_index([r for r in model if r.horizon == horizon and r.phi == phi])
    right = _record_index([r for r in baseline if r.horizon == horizon and r.phi == phi])
    if not left or left.keys() != right.keys():
        raise ValueError("paired comparisons require identical nonempty forecast keys")
    if len({r.model for r in left.values()}) != 1 or len({r.model for r in right.values()}) != 1:
        raise ValueError("paired comparison requires one model identity on each side")
    keys = sorted(left)
    for key in keys:
        a, b = left[key], right[key]
        if (
            a.y_true != b.y_true
            or a.instrument["gold_hash"] != b.instrument["gold_hash"]
            or a.instrument["normalization_hash"] != b.instrument["normalization_hash"]
            or a.population != b.population
            or {k: v for k, v in a.conditions.items() if k != "model"}
            != {k: v for k, v in b.conditions.items() if k != "model"}
        ):
            raise ValueError("paired comparisons require identical targets, data and validation scope")
    result = paired_bootstrap_rel_rmse(
        [left[k].y_true for k in keys],
        [left[k].y_pred for k in keys],
        [right[k].y_pred for k in keys],
        race_ids=[left[k].event_id for k in keys],
        stint_ids=[left[k].stint_id for k in keys],
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
    )
    return replace(
        result,
        conditions={
            **result.conditions,
            "horizon_actual_laps": horizon,
            "phi": phi,
            "validation_only": left[keys[0]].conditions["validation_only"],
        },
        instrument={
            **result.instrument,
            "model": sorted({r.model for r in left.values()}),
            "baseline": sorted({r.model for r in right.values()}),
            "gold_hashes": sorted({r.instrument["gold_hash"] for r in left.values()}),
        },
    )
