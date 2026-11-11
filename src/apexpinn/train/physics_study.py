"""Declared T13 jobs: inner-only tuning, validation scales, separate final fits and shared evaluation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from apexpinn.cloud.artifacts import digest, identity, write_json
from apexpinn.cloud.session import check_session
from apexpinn.contracts import Decisions
from apexpinn.data.gold import build_gold, fit_normalization
from apexpinn.data.guards import batch_from_gold
from apexpinn.eval.metrics import rmse
from apexpinn.eval.prepared import prepare_evaluation, run_prepared
from apexpinn.eval.protocol import (
    EvaluationFold,
    ProtocolConfig,
    inner_circuit_split,
    score_forecasts,
    summarize_origins,
)
from apexpinn.models.ensemble import NoWindowPhysics, PhysicsEnsemble, PhysicsMember
from apexpinn.models.ode_model import ODEModel
from apexpinn.models.pinn import PINN
from apexpinn.physics.identifiability import sample_prior_physics
from apexpinn.train.baseline_study import frame_digest
from apexpinn.train.cloud_checkpoint import CheckpointError, require_remote_gpu, train_remote, tree_digest
from apexpinn.train.run_meta import run_metadata
from apexpinn.train.trainer import LossWeights, TrainConfig

VARIANTS = ("main", "no_thermal", "no_wear", "constant_q", "no_window")


def configurations(method: dict, model: str) -> list[dict]:
    if model == "M1":
        values = [
            {"learning_rate": rate, "physics_weight": weight, "n_colloc": points}
            for rate in method[model]["learning_rates"]
            for weight in method[model]["physics_weights"]
            for points in method[model]["collocation_points"]
        ]
    elif model == "M2":
        low, high, count = method[model]["learning_rate_logspace"]
        values = [{"learning_rate": float(v)} for v in np.geomspace(low, high, count)]
    else:
        raise ValueError("model must be M1 or M2")
    if len(values) != 20 or len({identity(v) for v in values}) != 20:
        raise ValueError("exactly 20 unique configurations required")
    return values


def make_model(method: dict, name: str, parameters: dict, variant: str, seed: int):
    spec = method[name]
    key = jax.random.key(seed)
    physics = sample_prior_physics(jax.random.fold_in(key, 101))
    if variant == "no_window":
        physics = NoWindowPhysics(
            physics.raw_k, physics.raw_rho, physics.raw_gamma, physics.raw_h_opt, physics.raw_kappa
        )
    cfg = TrainConfig(
        n_steps=spec["steps"],
        learning_rate=parameters["learning_rate"],
        n_warm=spec["warm_steps"],
        n_ramp=spec["ramp_steps"],
        n_colloc=parameters.get("n_colloc", 32),
        float64=True,
    )
    weight = parameters.get("physics_weight", 1.0)
    weights = LossWeights(
        thermal=0.0 if variant == "no_thermal" else weight,
        wear=0.0 if variant == "no_wear" else weight,
        mono=0.0 if variant == "no_wear" else 0.1,
    )
    if name == "M1":
        return PINN.init(
            jax.random.fold_in(key, 1),
            physics=physics,
            train_cfg=cfg,
            weights=weights,
            width=spec["width"],
            depth=spec["depth"],
            latent_mode=spec["latent_mode"],
        )
    if variant != "main":
        raise ValueError("declared ablations apply to M1; M2 remains the structural comparator")
    return ODEModel.init(
        physics=physics, train_cfg=cfg, weights=weights, rtol=spec["rtol"], atol=spec["atol"]
    )


def batch_for_variant(batch, variant: str, constant: float):
    if variant != "constant_q":
        return batch
    return eqx.tree_at(
        lambda b: (b.q, b.ctx), batch, (jnp.full_like(batch.q, constant), batch.ctx.at[:, -1].set(constant))
    )


def fit_once(
    method: dict, name: str, parameters: dict, variant: str, seed: int, batch, folder: Path, run_id: dict
):
    """Resume only this fit's full state; failed/completed slots are immutable."""
    require_remote_gpu()
    check_session()
    template = make_model(method, name, parameters, variant, seed)
    expected = {
        **run_id,
        "model": name,
        "parameters": parameters,
        "variant": variant,
        "seed": seed,
        "batch_sha256": tree_digest(batch),
    }
    contract = {
        "evidence": "Observed",
        "quantity": "fit status and recorded wall-clock seconds",
        "population": {"batch_sha256": expected["batch_sha256"]},
        "conditions": {"model": name, "variant": variant, "seed": seed, "parameters": parameters},
        "instrument": {"clock": "time.perf_counter", "run_identity": run_id},
    }
    status_path = folder / "fit.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else None
    if previous and previous["identity"] != expected:
        raise CheckpointError("fit identity changed; use a new run directory")
    if previous and previous["status"] == "failed":
        raise RuntimeError("fit previously failed; no unbudgeted retry")
    if previous and previous["status"] == "finished":
        if digest(folder / "model.eqx") != previous["model_sha256"]:
            raise CheckpointError("saved fitted-model hash mismatch")
        return eqx.tree_deserialise_leaves(folder / "model.eqx", template), previous
    write_json(status_path, {**contract, "identity": expected, "status": "running"})
    started = time.perf_counter()
    try:
        fitted, trace = train_remote(
            template,
            batch,
            jax.random.key(seed),
            folder,
            expected,
            checkpoint_every=method["checkpoint_every"],
        )
        eqx.tree_serialise_leaves(folder / "model.eqx", fitted)
        record = {
            **contract,
            "identity": expected,
            "status": "finished",
            "elapsed_s": trace["recorded_elapsed_s"],
            "model_sha256": digest(folder / "model.eqx"),
            "state_sha256": tree_digest(fitted),
            "trace": trace,
        }
        write_json(status_path, record)
        return fitted, record
    except (CheckpointError, OSError, KeyboardInterrupt, SystemExit):
        # Transport/runtime interruption remains running. The same slot resumes its committed state.
        raise
    except Exception as exc:
        write_json(
            status_path,
            {
                **contract,
                "identity": expected,
                "status": "failed",
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def save_jsonl(path: Path, rows: list) -> None:
    temporary = path.with_suffix(".pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(asdict(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate_member(model, gold, stats, prepared, cfg, constant_q):
    check_session()
    member = PhysicsMember(model, constant_q)
    data = EvaluationFold(gold, stats, frame_digest(gold), member.state_digest(), prepared[0])
    return run_prepared(member, data, cfg, prepared[1], audit=[], horizons=(5,))


def run_job(
    silver: pd.DataFrame,
    decisions: Decisions,
    method: dict,
    root: Path,
    name: str,
    fold: str,
    phi_label: str,
    variant: str,
    run_id: dict,
) -> dict:
    require_remote_gpu()
    job = root / "jobs" / f"{name}_{fold}_{phi_label}_{variant}"
    result_path = job / "job.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["run_identity"] != run_id:
            raise ValueError("job identity changed")
        for filename, expected in result["files"].items():
            if digest(job / filename) != expected:
                raise ValueError("completed job payload changed")
        return result
    job.mkdir(parents=True, exist_ok=True)
    split = inner_circuit_split(decisions, fold)
    inner_stats = fit_normalization(
        silver, decisions, test_fold=fold, training_circuits=split.training_circuits
    )
    inner_gold = build_gold(silver, inner_stats, decisions)
    inner_batch = batch_from_gold(inner_gold, inner_stats)
    phi = float(silver.phi.iloc[0])
    cfg = ProtocolConfig(decisions, name if variant == "main" else f"{name}/{variant}", phi)
    static = EvaluationFold(
        inner_gold, inner_stats, frame_digest(inner_gold), "input-preparation", split.validation_fold
    )
    inner_plan = prepare_evaluation(fold, static, cfg, horizons=(5,))
    prepared = (split.validation_fold, inner_plan)
    seed_base = (
        method["seed"]
        + 10000 * tuple(decisions.folds).index(fold)
        + 100 * ("low", "mid", "high").index(phi_label)
    )
    tuning = []
    if variant == "main":
        for trial, parameters in enumerate(configurations(method, name)):
            path = job / "trials" / f"trial_{trial:02d}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["parameters"] != parameters or row["run_identity"] != run_id:
                    raise ValueError("completed trial identity mismatch")
                tuning.append(row)
                continue
            started = time.perf_counter()
            row = {
                "evidence": "Observed",
                "quantity": "primary-horizon inner RMSE and trial wall-clock seconds",
                "population": {"validation_circuits": list(split.validation_circuits)},
                "conditions": {"model": name, "fold": fold, "phi": phi, "horizon": 5},
                "instrument": {"clock": "time.perf_counter", "fit_folder": str(job / "trial_fits")},
                "trial": trial,
                "seed": seed_base + trial,
                "parameters": parameters,
                "run_identity": run_id,
            }
            fit_folder = job / "trial_fits" / f"trial_{trial:02d}"
            try:
                fitted, fit = fit_once(
                    method, name, parameters, variant, seed_base + trial, inner_batch, fit_folder, run_id
                )
                records = evaluate_member(fitted, inner_gold, inner_stats, prepared, cfg, None)
                score = rmse([r.y_true for r in records], [r.y_pred for r in records])
                row.update(status="finished", rmse_primary_s=score)
                fit_seconds = fit["elapsed_s"]
            except (CheckpointError, OSError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}", rmse_primary_s=None)
                fit_record = fit_folder / "fit.json"
                fit_seconds = (
                    json.loads(fit_record.read_text()).get("elapsed_s") if fit_record.exists() else None
                )
            row.update(
                elapsed_s=time.perf_counter() - started,
                fit_recorded_s=fit_seconds,
                clock="time.perf_counter, current invocation fit+primary forecast",
                interruption_overhead="Unknown when an uncheckpointed segment was lost",
            )
            write_json(path, row)
            tuning.append(row)
        successful = [r for r in tuning if r["status"] == "finished"]
        if not successful:
            raise RuntimeError("all 20 trials failed; fixed budget exhausted")
        chosen = min(successful, key=lambda r: (r["rmse_primary_s"], r["trial"]))["parameters"]
        write_json(
            job / "selection.json",
            {"parameters": chosen, "rule": "minimum inner h5 RMSE; trial index tie-break"},
        )
    else:
        source = root / "jobs" / f"{name}_{fold}_{phi_label}_main" / "selection.json"
        if not source.exists():
            raise ValueError("run main tuning before its fixed-hyperparameter ablations")
        chosen = json.loads(source.read_text())["parameters"]
    outer_stats = fit_normalization(silver, decisions, test_fold=fold)
    outer_gold = build_gold(silver, outer_stats, decisions)
    outer_batch = batch_from_gold(outer_gold, outer_stats)
    inner_constant, outer_constant = inner_stats.training_q_mean, outer_stats.training_q_mean
    inner_batch = batch_for_variant(inner_batch, variant, inner_constant)
    outer_batch = batch_for_variant(outer_batch, variant, outer_constant)
    members, calibrations, finals = [], [], []
    for member in range(5):
        member_seed = 1_000_000 + seed_base + member
        # Fixed-configuration validation fits estimate scales only, never select hyperparameters.
        calibrated, calibration = fit_once(
            method,
            name,
            chosen,
            variant,
            member_seed,
            inner_batch,
            job / "calibration" / f"member_{member}",
            run_id,
        )
        calibration_records = evaluate_member(
            calibrated,
            inner_gold,
            inner_stats,
            prepared,
            cfg,
            inner_constant if variant == "constant_q" else None,
        )
        noise = max(
            rmse([r.y_true for r in calibration_records], [r.y_pred for r in calibration_records]), 1e-6
        )
        fitted, final = fit_once(
            method,
            name,
            chosen,
            variant,
            member_seed,
            outer_batch,
            job / "final" / f"member_{member}",
            run_id,
        )
        fitted = eqx.tree_at(lambda m: m.noise_s, fitted, jnp.asarray(noise))
        members.append(PhysicsMember(fitted, outer_constant if variant == "constant_q" else None))
        calibrations.append(
            {"member": member, "elapsed_s": calibration["elapsed_s"], "validation_scale_s": noise}
        )
        finals.append(
            {
                "member": member,
                "seed": member_seed,
                "elapsed_s": final["elapsed_s"],
                "state_sha256": tree_digest(fitted),
            }
        )
        eqx.tree_serialise_leaves(job / "final" / f"member_{member}" / "calibrated_model.eqx", fitted)
    if len({r["state_sha256"] for r in finals}) != 5:
        raise ValueError("physics ensemble members are not distinct")
    ensemble = PhysicsEnsemble(tuple(members))
    data = EvaluationFold(outer_gold, outer_stats, frame_digest(outer_gold), ensemble.state_digest())
    plan = prepare_evaluation(fold, replace(data, model_hash="input-preparation"), cfg)
    audit = []
    check_session()
    records = run_prepared(
        ensemble,
        data,
        cfg,
        plan,
        audit=audit,
        checkpoint_dir=job / "forecast_checkpoints"
        if os.environ.get("APEXPINN_REMOTE_RUNTIME") == "kaggle"
        else None,
    )
    metrics = score_forecasts(records)
    save_jsonl(job / "forecasts.jsonl", records)
    save_jsonl(job / "origin_audit.jsonl", audit)
    from apexpinn.eval.physics_report import job_diagnostics

    check_session()
    diagnostics = job_diagnostics(ensemble, outer_gold, records, method)
    result = {
        "schema": "apexpinn.t13_job/v1",
        "evidence": "Observed",
        "run_identity": run_id,
        "quantity": "held-out physics-model forecasts and diagnostics",
        "population": {"fold": fold, "kind": "development"},
        "conditions": {
            "model": name,
            "variant": variant,
            "phi": phi,
            "phi_label": phi_label,
            "primary_horizon": 5,
            "ensemble_members": 5,
            "normalization_training_circuits": list(outer_stats.training_circuits),
        },
        "instrument": {
            "runtime": run_metadata(seeds=[r["seed"] for r in finals], data_hash=frame_digest(outer_gold)),
            "inner_training_circuits": list(inner_stats.training_circuits),
            "inner_validation_circuits": list(split.validation_circuits),
        },
        "trials": tuning,
        "selection": chosen,
        "calibration_fits": calibrations,
        "final_fits": finals,
        "metrics": [asdict(m) for m in metrics],
        "origin_metrics": [asdict(m) for m in summarize_origins(audit)],
        "diagnostics": diagnostics,
        "files": {
            p.relative_to(job).as_posix(): digest(p)
            for p in sorted(job.rglob("*"))
            if p.is_file()
            and (
                p.name in ("forecasts.jsonl", "origin_audit.jsonl", "fit.json", "selection.json")
                or (p.suffix == ".eqx" and "checkpoints" not in p.parts)
            )
        },
        "G2": "NOT ESTABLISHED: job only, full sweep and user P2 review remain separate",
    }
    write_json(result_path, result)
    return result


def read_method(path: Path) -> dict:
    method = yaml.safe_load(path.read_text())
    if (
        (method["method_id"], method["training_location"])
        not in (
            ("apexpinn.t13_cloud/v1", "colab_gpu_only"),
            ("apexpinn.t13_sagemaker/v1", "sagemaker_l4_training_job_only"),
            ("apexpinn.t13_studio/v1", "sagemaker_studio_l4_only"),
            ("apexpinn.t13_ec2/v1", "ec2_l4_only"),
            ("apexpinn.t13_kaggle/v1", "kaggle_single_t4_only"),
        )
        or method["ensemble_members"] != 5
        or method["tuning_trials"] != 20
        or method["horizons"] != [1, 3, 5, 10]
        or method["primary_horizon"] != 5
        or method["float64"] is not True
        or method["uncertainty"] != "exact_equal_weight_Gaussian_mixture_no_maximum_noise_floor"
    ):
        raise ValueError("unsupported method identity/budget")
    for name in ("M1", "M2"):
        configurations(method, name)
    return method
