"""Replay the existing protocol's immutable inputs across candidates without rebuilding Pandas slices.

The original runner is the only source of eligibility, histories, labels and audit decisions. Its
temporary zero outputs are discarded, never scored, and are not fitted model predictions.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from apexpinn.contracts import PredictionOutput
from apexpinn.eval.protocol import (
    EvaluationFold,
    ForecastRecord,
    OriginAudit,
    ProtocolConfig,
    _component_scales,
    run_protocol,
)
from apexpinn.models.stints import Forecaster, Scenario, StintHistory


def _identity(data: EvaluationFold, cfg: ProtocolConfig) -> str:
    return json.dumps(
        (
            asdict(cfg.decisions),
            cfg.phi,
            cfg.validation_only,
            data.gold_hash,
            asdict(data.normalization),
            data.inner_validation_fold,
        ),
        sort_keys=True,
    )


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


class _CaptureInputs:
    def __init__(self):
        self.inputs = []

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        frozen = replace(history, tau=_readonly(history.tau), y=_readonly(history.y), q=_readonly(history.q))
        self.inputs.append((frozen, scenario))
        return PredictionOutput(np.zeros(horizon), np.zeros((1, horizon)), np.ones(horizon), None, None)


@dataclass(frozen=True)
class PreparedEvaluation:
    identity: str
    origins: tuple[OriginAudit, ...]
    inputs: dict[tuple, tuple[ForecastRecord, StintHistory, Scenario]]
    preparation_s: float
    horizons: tuple[int, ...]


def prepare_evaluation(
    fold: str, data: EvaluationFold, cfg: ProtocolConfig, *, horizons: tuple[int, ...] | None = None
) -> PreparedEvaluation:
    """Construct one model-independent population through the original runner; never fit or score."""
    started = time.perf_counter()
    capture = _CaptureInputs()
    audit = []
    templates = run_protocol(capture, {fold: data}, cfg, audit=audit, horizons=horizons)
    inputs = {
        record.key: (record, history, scenario)
        for record, (history, scenario) in zip(templates, capture.inputs, strict=True)
    }
    selected = tuple(cfg.decisions.horizons) if horizons is None else horizons
    return PreparedEvaluation(
        _identity(data, cfg), tuple(audit), inputs, time.perf_counter() - started, selected
    )


def run_prepared(
    forecaster: Forecaster,
    data: EvaluationFold,
    cfg: ProtocolConfig,
    prepared: PreparedEvaluation,
    *,
    audit: list[OriginAudit],
    horizons: tuple[int, ...] | None = None,
    checkpoint_dir: Path | None = None,
) -> list[ForecastRecord]:
    """Replay identical inputs/endpoint labels; optional declared horizon subset is for inner tuning."""
    if _identity(data, cfg) != prepared.identity:
        raise ValueError("prepared input identity differs from supplied config/data")
    selected = prepared.horizons if horizons is None else horizons
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(prepared.horizons):
        raise ValueError("requested horizons must be a unique declared subset")
    records = []
    for origin in prepared.origins:
        if origin.horizon not in selected:
            continue
        conditions = {**origin.conditions, "model": cfg.model_name}
        instrument = {
            **origin.instrument,
            "model_hash": data.model_hash,
            "input_execution": "immutable replay/v1",
        }
        if origin.eligible:
            key = (origin.fold, origin.event_id, origin.stint_id, origin.t0, origin.horizon, cfg.phi)
            template, history, scenario = prepared.inputs[key]
            cached = None
            cache_path = None
            if checkpoint_dir is not None:
                from apexpinn.cloud.artifacts import identity
                from apexpinn.cloud.session import check_session

                check_session()
                cache_key = identity([key, data.model_hash, prepared.identity, cfg.model_name])
                cache_path = checkpoint_dir / f"{cache_key}.json"
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text())
                    if cached["key"] != cache_key or identity(cached["record"]) != cached["sha256"]:
                        raise ValueError("forecast checkpoint identity/payload mismatch")
            if cached is not None:
                records.append(ForecastRecord(**cached["record"]))
                audit.append(replace(origin, conditions=conditions, instrument=instrument))
                continue
            output = forecaster.forecast(history, origin.horizon, scenario)
            if not isinstance(output, PredictionOutput) or output.mean_s.shape != (origin.horizon,):
                raise ValueError("forecaster returned an invalid PredictionOutput/horizon")
            records.append(
                replace(
                    template,
                    model=cfg.model_name,
                    y_pred=float(output.mean_s[-1]),
                    sigma_s=float(output.sigma_s[-1]),
                    member_means_s=tuple(float(v) for v in output.member_means_s[:, -1]),
                    member_sigma_s=_component_scales(output, origin.horizon),
                    conditions=conditions,
                    instrument=instrument,
                )
            )
            if cache_path is not None:
                from apexpinn.cloud.artifacts import write_json

                row = asdict(records[-1])
                write_json(cache_path, {"key": cache_key, "record": row, "sha256": identity(row)})
        audit.append(replace(origin, conditions=conditions, instrument=instrument))
    return records
