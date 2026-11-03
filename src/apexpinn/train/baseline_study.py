"""Baseline tuning/ensemble execution with explicit trial and runtime accounting (PLAN T12/O7)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd

from apexpinn.contracts import Decisions
from apexpinn.data import manifest as mf
from apexpinn.data.gold import build_gold, fit_normalization
from apexpinn.data.guards import batch_from_gold
from apexpinn.eval.metrics import rmse
from apexpinn.eval.prepared import PreparedEvaluation, prepare_evaluation, run_prepared
from apexpinn.eval.protocol import (
    EvaluationFold,
    ProtocolConfig,
    inner_circuit_split,
    paired_comparison,
    score_forecasts,
    summarize_origins,
)
from apexpinn.models.baseline_kernels import CompiledBaseline
from apexpinn.models.baselines import B0, B1, B2, B3, BaselineEnsemble
from apexpinn.train.run_meta import run_metadata

MODEL_NAMES = ("B0", "B1", "B2", "B3")


def candidate_configs(name: str) -> tuple[dict[str, Any], ...]:
    """Declared unique configurations; B0/B2 use O7's finite-space exception, without trial padding."""
    if name in ("B0", "B2"):
        return ({},)
    if name == "B1":
        return tuple(
            {"n_estimators": trees, "num_leaves": leaves, "learning_rate": 0.05}
            for trees in (50, 100, 150, 200)
            for leaves in (7, 15, 31, 63, 127)
        )
    if name == "B3":
        return tuple({"learning_rate": float(rate)} for rate in np.geomspace(0.001, 0.1, 20))
    raise ValueError(f"unknown baseline {name!r}")


def make_baseline(name: str, params: dict[str, Any], key: jax.Array, *, mlp_steps: int = 500) -> Any:
    if name == "B0":
        return B0(**params)
    if name == "B1":
        return B1(**params)
    if name == "B2":
        return B2(**params)
    if name == "B3":
        return B3.init(key, n_steps=mlp_steps, **params)
    raise ValueError(f"unknown baseline {name!r}")


def frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class TrialRecord:
    model: str
    outer_fold: str
    phi: float
    trial: int
    parameters: dict[str, Any]
    seed: int
    status: str
    elapsed_s: float
    rmse_primary_s: float | None
    error: str | None
    quantity: dict[str, Any]
    population: dict[str, Any]
    conditions: dict[str, Any]
    instrument: dict[str, Any]


def tune_baseline(
    silver: pd.DataFrame,
    decisions: Decisions,
    *,
    model_name: str,
    outer_fold: str,
    seed: int,
    validation_only: bool,
    trials: list[TrialRecord],
    mlp_steps: int = 500,
    on_trial: Callable[[TrialRecord], None] | None = None,
    factory: Callable[..., Any] = make_baseline,
    completed: Sequence[TrialRecord] = (),
    prepared_cache: dict[str, PreparedEvaluation] | None = None,
    execution_mode: str = "prepared/v2",
) -> dict[str, Any]:
    """Evaluate each declared configuration once; failed fits count, no extra retries or hidden search."""
    if decisions.status == "draft" and not validation_only:
        raise ValueError("draft config requires validation-only tuning before any model fit")
    budget = int(decisions.ensemble["tuning_budget"]["per_model"])
    configurations = candidate_configs(model_name)
    if budget != 20 or len(configurations) > budget:
        raise ValueError("baseline protocol requires the declared 20-trial budget")
    if len({json.dumps(p, sort_keys=True) for p in configurations}) != len(configurations):
        raise ValueError("trial configurations must be unique")
    split = inner_circuit_split(decisions, outer_fold)
    stats = fit_normalization(
        silver, decisions, test_fold=outer_fold, training_circuits=split.training_circuits
    )
    gold = build_gold(silver, stats, decisions)
    train = batch_from_gold(gold, stats, k_min=decisions.k_min)
    source_hash = frame_digest(gold)
    phi = float(gold.phi.iloc[0])
    prior = {r.trial: r for r in completed if r.model == model_name and r.outer_fold == outer_fold}
    if len(prior) != sum(r.model == model_name and r.outer_fold == outer_fold for r in completed):
        raise ValueError("duplicate resume trial IDs")
    for index, record in prior.items():
        if (
            not 0 <= index < len(configurations)
            or record.parameters != configurations[index]
            or record.seed != seed + index
            or record.phi != phi
            or record.conditions["mlp_steps"] != mlp_steps
            or record.conditions["validation_only"] != validation_only
            or record.conditions["primary_horizon"] != decisions.primary_horizon
            or record.conditions["trial_budget"] != budget
            or record.conditions["declared_unique_configurations"] != len(configurations)
            or record.instrument["gold_hash"] != source_hash
            or record.instrument["normalization_training_hash"] != stats.training_rows_sha256
            or record.status not in ("finished", "failed")
            or not np.isfinite(record.elapsed_s)
            or record.elapsed_s < 0
            or (
                record.status == "finished"
                and (record.rmse_primary_s is None or not np.isfinite(record.rmse_primary_s))
            )
        ):
            raise ValueError("resume trial identity/config/data differs from declared run")
    cfg = ProtocolConfig(decisions, model_name, phi, validation_only=validation_only)
    static_data = EvaluationFold(gold, stats, source_hash, "input-preparation", split.validation_fold)
    cache = {} if prepared_cache is None else prepared_cache
    cache_key = f"inner/{outer_fold}/{phi}"
    if len(prior) < len(configurations) and cache_key not in cache:
        cache[cache_key] = prepare_evaluation(
            outer_fold, static_data, cfg, horizons=(decisions.primary_horizon,)
        )
    successful = []
    for trial, params in enumerate(configurations):
        if trial in prior:
            record = prior[trial]
            trials.append(record)
            if on_trial is not None:
                on_trial(record)
            if record.status == "finished":
                successful.append((record.rmse_primary_s, trial, params))
            continue
        trial_seed = seed + trial
        started = time.perf_counter()
        score = None
        error = None
        status = "failed"
        try:
            key = jax.random.key(trial_seed)
            model = factory(model_name, params, key, mlp_steps=mlp_steps).fit(train, key)
            data = EvaluationFold(gold, stats, source_hash, model.state_digest(), split.validation_fold)
            records = run_prepared(
                model, data, cfg, cache[cache_key], audit=[], horizons=(decisions.primary_horizon,)
            )
            if not records:
                raise ValueError("no valid primary-horizon inner validation score")
            score = rmse([r.y_true for r in records], [r.y_pred for r in records])
            status = "finished"
            successful.append((score, trial, params))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        record = TrialRecord(
            model_name,
            outer_fold,
            phi,
            trial,
            dict(params),
            trial_seed,
            status,
            elapsed,
            score,
            error,
            quantity={"metric": "inner primary-horizon RMSE", "unit": "s"},
            population={
                "training_circuits": list(stats.training_circuits),
                "validation_circuits": list(split.validation_circuits),
                "outer_fold": outer_fold,
            },
            conditions={
                "trial_budget": budget,
                "declared_unique_configurations": len(configurations),
                "primary_horizon": decisions.primary_horizon,
                "validation_only": validation_only,
                "mlp_steps": mlp_steps,
                "failed_trials_count": True,
            },
            instrument={
                "clock": "time.perf_counter",
                "clock_scope": "factory + fit + primary-horizon forecasting/RMSE; shared inputs separate",
                "input_execution": "immutable replay/v1",
                "numeric_execution": execution_mode,
                "prepared_inputs_key": cache_key,
                "gold_hash": source_hash,
                "normalization_training_hash": stats.training_rows_sha256,
            },
        )
        trials.append(record)
        if on_trial is not None:
            on_trial(record)
    if not successful:
        raise ValueError(f"all {model_name} trial configurations failed; records retained")
    return dict(min(successful, key=lambda item: (item[0], item[1]))[2])


def fixture_silver(decisions: Decisions, phi: float = 0.06) -> pd.DataFrame:
    """Small crafted fixture spanning the declared groups; event names are labels, not real race data."""
    frames = []
    for index, (fold, circuits) in enumerate(decisions.folds.items()):
        event = next(e for e in decisions.dev_events if e.split("_", 1)[1] in circuits)
        season, circuit = event.split("_", 1)
        lap = np.arange(1, 19)
        y = 90 + index + (0.04 + 0.005 * index) * lap + 0.02 * np.sin(lap)
        frames.append(
            pd.DataFrame(
                {
                    "session_id": event,
                    "season": int(season),
                    "circuit_id": circuit,
                    "driver_id": "fixture",
                    "stint_id": event + ":fixture:1",
                    "lap_number": lap,
                    "race_lap": lap,
                    "stint_lap": lap,
                    "race_length": 18,
                    "split_fold": fold,
                    "dataset_split": "development",
                    "event_is_dry": True,
                    "is_scored_lap": lap > 1,
                    "exclusion_reason": np.where(lap == 1, "race_lap_1", ""),
                    "compound_c": "C3",
                    "source_version": "crafted_baseline_fixture/v1",
                    "phi": phi,
                    "obs_lap_time_s": y + phi * (18 - lap),
                    "obs_tyre_life": lap + 2.0,
                    "obs_fresh_tyre": False,
                    "obs_track_temp_c": 30.0 + index,
                    "obs_speed_st_kmh": 300.0,
                    "proxy_y": y,
                    "proxy_energy_index": 15 + index * 1.5 + 0.01 * lap + 0.1 * np.sin(lap),
                    "proxy_brake_fraction": 0.25,
                    "proxy_brake_fraction_weighted": 0.2,
                    "proxy_telemetry_missing": False,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_study(
    silver: pd.DataFrame,
    decisions: Decisions,
    *,
    out: Path,
    report: Path,
    models: Sequence[str] = MODEL_NAMES,
    folds: Sequence[str] | None = None,
    seed: int = 0,
    validation_only: bool = False,
    source_kind: str = "development",
    coverage_complete: bool = False,
    mlp_steps: int = 500,
    resume_trials: Sequence[TrialRecord] = (),
    resume_provenance: dict[str, Any] | None = None,
    compiled_kernels: bool = False,
) -> dict[str, Any]:
    """Full declared baseline search then five final fits; fixture output cannot establish G1."""
    selected_folds = tuple(decisions.folds) if folds is None else tuple(folds)
    if not models or not set(models) <= set(MODEL_NAMES) or len(set(models)) != len(models):
        raise ValueError("models must be unique baseline names")
    if (
        not selected_folds
        or not set(selected_folds) <= set(decisions.folds)
        or len(set(selected_folds)) != len(selected_folds)
    ):
        raise ValueError("folds must be unique configured groups")
    if not validation_only and (
        decisions.status != "frozen" or not coverage_complete or source_kind != "development"
    ):
        raise ValueError("G0 coverage/config freeze required; validation-only fixtures do not establish G1")
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory already contains records; choose a new run directory")
    if report.suffix.lower() != ".md":
        raise ValueError("report must be a Markdown file, separate from study JSON records")
    if report.exists() and report.read_text(encoding="utf-8").splitlines()[:1] != [
        "# Baseline evaluation report"
    ]:
        raise ValueError("refusing to overwrite an unrelated existing report")
    if not silver.dataset_split.eq("development").all():
        raise ValueError("temporal holdout is forbidden in baseline development")
    member_count = int(decisions.ensemble["ensemble_size"])
    if member_count != 5:
        raise ValueError("baseline study requires the declared five members")
    input_hash = frame_digest(silver)
    runtime_metadata = run_metadata(seeds=[seed], data_hash=input_hash)
    if any(r.model not in models or r.outer_fold not in selected_folds for r in resume_trials):
        raise ValueError("resume trials fall outside requested models/folds")
    if len({(r.model, r.outer_fold, r.trial) for r in resume_trials}) != len(resume_trials):
        raise ValueError("duplicate resume trial IDs")
    out.mkdir(parents=True, exist_ok=True)
    source_names = (
        "models/baselines.py",
        "models/baseline_kernels.py",
        "train/baseline_study.py",
        "eval/protocol.py",
        "eval/prepared.py",
        "data/guards.py",
        "eval/baseline_report.py",
        "contracts.py",
        "models/stints.py",
        "data/gold.py",
        "train/run_meta.py",
    )
    source_hashes = {}
    for name in source_names:
        path = Path(__file__).parents[1] / name
        source_hashes[name] = mf.sha256_file(path)
        snapshot = out / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    records = []
    audits = []
    trials = []
    final_fits = []
    selections = {}
    failures = {}
    prepared_cache = {}
    execution_mode = "compiled/v3" if compiled_kernels else "prepared/v2"

    def build_candidate(name, params, key, *, mlp_steps):
        model = make_baseline(name, params, key, mlp_steps=mlp_steps)
        return CompiledBaseline(model) if compiled_kernels and name != "B0" else model

    def save_trial(record):
        folder = out / "trials"
        folder.mkdir(exist_ok=True)
        path = folder / f"{record.outer_fold}_{record.model}_{record.trial:02d}.json"
        path.write_text(
            json.dumps(asdict(record), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )

    for f_index, fold in enumerate(selected_folds):
        stats = fit_normalization(silver, decisions, test_fold=fold)
        gold = build_gold(silver, stats, decisions)
        batch = batch_from_gold(gold, stats, k_min=decisions.k_min)
        outer_hash = frame_digest(gold)
        static_outer = EvaluationFold(gold, stats, outer_hash, "input-preparation")
        prepared_cache[f"outer/{fold}"] = prepare_evaluation(
            fold,
            static_outer,
            ProtocolConfig(decisions, "input-preparation", float(silver.phi.iloc[0]), validation_only),
        )
        for m_index, name in enumerate(models):
            run_seed = seed + 10000 * f_index + 100 * m_index
            try:
                params = tune_baseline(
                    silver,
                    decisions,
                    model_name=name,
                    outer_fold=fold,
                    seed=run_seed,
                    validation_only=validation_only,
                    trials=trials,
                    mlp_steps=mlp_steps,
                    on_trial=save_trial,
                    completed=resume_trials,
                    prepared_cache=prepared_cache,
                    factory=build_candidate,
                    execution_mode=execution_mode,
                )
                selections[f"{fold}/{name}"] = params
                members = []
                for member in range(member_count):
                    member_seed = seed + 1_000_000 + 1000 * f_index + 100 * m_index + member
                    started = time.perf_counter()
                    key = jax.random.key(member_seed)
                    fitted = None
                    fit_error = None
                    try:
                        fitted = build_candidate(name, params, key, mlp_steps=mlp_steps).fit(batch, key)
                    except Exception as exc:
                        fit_error = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        elapsed = time.perf_counter() - started
                        final_fits.append(
                            {
                                "model": name,
                                "fold": fold,
                                "member": member,
                                "seed": member_seed,
                                "elapsed_s": elapsed,
                                "state_sha256": fitted.state_digest() if fitted is not None else None,
                                "status": "finished" if fitted is not None else "failed",
                                "error": fit_error,
                                "quantity": {"metric": "final member fit time", "unit": "s"},
                                "population": {"outer_training_circuits": list(stats.training_circuits)},
                                "conditions": {
                                    "validation_only": validation_only,
                                    "not_a_tuning_trial": True,
                                },
                                "instrument": {"clock": "time.perf_counter"},
                            }
                        )
                        fit_path = out / "final_fits" / f"{fold}_{name}_{member}.json"
                        fit_path.parent.mkdir(exist_ok=True)
                        fit_path.write_text(json.dumps(final_fits[-1], indent=2, sort_keys=True) + "\n")
                    members.append(fitted)
                ensemble = BaselineEnsemble(tuple(members))
                cfg = ProtocolConfig(decisions, name, float(silver.phi.iloc[0]), validation_only)
                data = EvaluationFold(gold, stats, outer_hash, ensemble.state_digest())
                records.extend(
                    run_prepared(ensemble, data, cfg, prepared_cache[f"outer/{fold}"], audit=audits)
                )
            except Exception as exc:
                failures[f"{fold}/{name}"] = f"{type(exc).__name__}: {exc}"
    metrics = score_forecasts(records) if records else []
    origin_metrics = summarize_origins(audits) if audits else []
    # Reference choice uses inner scores only, before any outer score is consulted.
    inner_scores = {name: [] for name in models}
    for fold in selected_folds:
        for name in models:
            values = [
                r.rmse_primary_s
                for r in trials
                if r.outer_fold == fold and r.model == name and r.status == "finished"
            ]
            if values:
                inner_scores[name].append(min(values))
    comparable = [name for name in models if len(inner_scores[name]) == len(selected_folds)]
    reference = min(comparable, key=lambda n: (float(np.mean(inner_scores[n])), n)) if comparable else None
    classic = [name for name in comparable if name in ("B0", "B1", "B2")]
    classic_reference = min(classic, key=lambda n: (float(np.mean(inner_scores[n])), n)) if classic else None
    comparisons = []
    if reference is not None:
        for name in models:
            if name == reference:
                continue
            for horizon in decisions.horizons:
                try:
                    result = paired_comparison(
                        [r for r in records if r.model == name],
                        [r for r in records if r.model == reference],
                        horizon=horizon,
                        phi=float(silver.phi.iloc[0]),
                        seed=seed,
                    )
                    comparisons.append(
                        {"model": name, "reference": reference, "horizon": horizon, **asdict(result)}
                    )
                except ValueError as exc:
                    comparisons.append(
                        {
                            "model": name,
                            "reference": reference,
                            "horizon": horizon,
                            "status": "Unknown",
                            "reason": str(exc),
                            "quantity": {"metric": "paired_relative_rmse_reduction", "unit": "fraction"},
                            "population": {"folds": list(selected_folds)},
                            "conditions": {"validation_only": validation_only},
                            "instrument": {"method": "paired race-then-stint bootstrap/v1"},
                        }
                    )
    accepted = (
        not validation_only
        and not failures
        and coverage_complete
        and set(models) == set(MODEL_NAMES)
        and set(selected_folds) == set(decisions.folds)
    )
    receipt = {
        "schema": "apexpinn.baselines/v2",
        "evidence": "Observed",
        "quantity": "baseline evaluation and trial accounting",
        "population": {
            "kind": source_kind,
            "folds": list(selected_folds),
            "events": sorted(silver.session_id.unique()),
        },
        "conditions": {
            "validation_only": validation_only,
            "coverage_complete": coverage_complete,
            "phi": float(silver.phi.iloc[0]),
            "mlp_steps": mlp_steps,
            "ensemble_members": member_count,
        },
        "instrument": {
            "source_sha256": source_hashes,
            "execution": execution_mode + ": immutable inputs and primary-only inner RMSE",
            "compiled_kernels": compiled_kernels,
            "shared_input_preparation_s": {name: p.preparation_s for name, p in prepared_cache.items()},
            "resumed_trials": len(resume_trials),
            "resume_provenance": resume_provenance,
            "input_frame_sha256": input_hash,
            "runtime": runtime_metadata,
            "seed": seed,
            "versions": {
                name: importlib.metadata.version(name) for name in ("lightgbm", "numpy", "pandas", "scipy")
            },
            "jax_backend": jax.default_backend(),
            "float64_enabled": jax.config.x64_enabled,
        },
        "G1": "EVIDENCE READY FOR REVIEW" if accepted else "NOT ESTABLISHED",
        "selected_reference_by_inner_validation": reference,
        "selected_B0_B2_reference_by_inner_validation": classic_reference,
        "inner_scores": inner_scores,
        "selections": selections,
        "trials": [asdict(r) for r in trials],
        "final_member_fits": final_fits,
        "forecasts": [asdict(r) for r in records],
        "origin_audit": [asdict(r) for r in audits],
        "metrics": [asdict(r) for r in metrics],
        "origin_metrics": [asdict(r) for r in origin_metrics],
        "paired_comparisons": comparisons,
        "failures": failures,
    }
    path = out / "study.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    from apexpinn.eval.baseline_report import build_baseline_report

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_baseline_report(receipt, source=str(path)), encoding="utf-8")
    return receipt
