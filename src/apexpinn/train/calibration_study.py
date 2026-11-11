"""Separate clean-calibration pilot. Reuses existing cloud fits without modifying T13 v1."""

from __future__ import annotations

import importlib.metadata
import json
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from apexpinn.cloud.artifacts import digest, identity, write_json
from apexpinn.eval.calibration import apply_calibrator, fit_calibrator, split_plan


def read_spec(path: Path) -> dict:
    method = yaml.safe_load(path.read_text())
    if digest(Path(method["base_training_method"])) != method["base_method_sha256"]:
        raise ValueError("base training method differs from the declared frozen source")
    if (
        method["method_id"] != "apexpinn.calibration_clean/v1"
        or method["model"] != "M2"
        or method["tuning_trials"] != 20
        or method["ensemble_members"] != 5
        or method["final_fits"] != 5
        or method["extra_calibration_fits"] != 0
        or method["refit_after_calibration"] is not False
        or method["horizons"] != [1, 3, 5, 10]
        or method["primary_horizon"] != 5
        or method["float64"] is not True
        or method["training_location"] != "colab_gpu_only"
        or method["seed"] != 0
        or method["split"] != "outer_i_tuning_validation_i_plus_1_calibration_i_plus_2_cyclic"
        or method["raw_noise"] != "final_member_training_residual_RMSE"
        or method["calibrator"]
        != {
            "estimator": "weighted_empirical_90pct_interval_ratio",
            "target_level": 0.9,
            "weighting": "equal_event_then_equal_origin_within_horizon",
            "minimum_events_per_horizon": 2,
            "factor_floor": 1e-6,
            "transform": "affine_entire_equal_weight_mixture_about_unchanged_mean",
            "context_dependence": "none_in_v1",
        }
    ):
        raise ValueError("unsupported calibration method; declare a new version for changed settings")
    return method


def select_rows(frame, circuits):
    """Physically restrict labels passed to fitting or selection consumers."""
    if not frame.dataset_split.eq("development").all():
        raise ValueError("temporal holdout is forbidden in this calibration pilot")
    result = frame.loc[frame.circuit_id.isin(circuits)].copy()
    if result.empty:
        raise ValueError("empty declared data role")
    return result


def _remote_guard() -> None:
    from apexpinn.train.cloud_checkpoint import require_colab_gpu

    require_colab_gpu()


def _normalization_id(stats) -> str:
    return identity({k: v for k, v in asdict(stats).items() if k != "test_fold"})


def _predict(model, frame, stats, decisions, outer: str, role: str, horizons: tuple):
    from apexpinn.data.gold import build_gold
    from apexpinn.eval.prepared import prepare_evaluation, run_prepared
    from apexpinn.eval.protocol import EvaluationFold, ProtocolConfig
    from apexpinn.train.baseline_study import frame_digest

    gold = build_gold(frame, stats, decisions)
    cfg = ProtocolConfig(decisions, "M2/raw", stats.phi)
    data = EvaluationFold(gold, stats, frame_digest(gold), model.state_digest())
    plan = prepare_evaluation(stats.test_fold, data, cfg, horizons=horizons)
    audit = []
    records = run_prepared(model, data, cfg, plan, audit=audit)
    if not records:
        raise ValueError(f"no eligible forecasts for {role}")
    tagged = [
        replace(
            r,
            conditions={
                **r.conditions,
                "data_role": role,
                "outer_fold": outer,
                "calibration_method": "apexpinn.calibration_clean/v1",
            },
            instrument={**r.instrument, "normalization_numeric_sha256": _normalization_id(stats)},
        )
        for r in records
    ]
    return [asdict(r) for r in tagged], [asdict(r) for r in audit]


def _prediction_cache(folder: Path, stamp: dict, produce) -> list[dict]:
    """Commit predictions and their receipt together; reuse completed forecasts without replay."""
    if folder.exists():
        meta = json.loads((folder / "receipt.json").read_text())
        if meta["identity"] != stamp or any(
            digest(folder / name) != sha for name, sha in meta["files"].items()
        ):
            raise ValueError("forecast cache identity or payload changed")
        return [json.loads(line) for line in (folder / "forecasts.jsonl").open()]
    records, audit = produce()
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        stage = Path(tmp) / "complete"
        stage.mkdir()
        for name, values in (("forecasts.jsonl", records), ("audit.jsonl", audit)):
            with (stage / name).open("w") as stream:
                for row in values:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        write_json(
            stage / "receipt.json",
            {
                "identity": stamp,
                "files": {name: digest(stage / name) for name in ("forecasts.jsonl", "audit.jsonl")},
            },
        )
        stage.rename(folder)
    return records


def _run_trials(method, base, decisions, frame, stats, batch, plan, root, run_id, seed_base):
    from apexpinn.data.gold import build_gold
    from apexpinn.eval.metrics import rmse
    from apexpinn.eval.prepared import prepare_evaluation, run_prepared
    from apexpinn.eval.protocol import EvaluationFold, ProtocolConfig
    from apexpinn.models.ensemble import PhysicsMember
    from apexpinn.train.baseline_study import frame_digest
    from apexpinn.train.cloud_checkpoint import CheckpointError
    from apexpinn.train.physics_study import configurations, fit_once

    gold = build_gold(frame, stats, decisions)
    cfg = ProtocolConfig(decisions, "M2/tuning", stats.phi)
    data = EvaluationFold(gold, stats, frame_digest(gold), "preparation", plan["validation_fold"])
    prepared = prepare_evaluation(plan["outer_fold"], data, cfg, horizons=(5,))
    trials = []
    for trial, parameters in enumerate(configurations(base, "M2")):
        path = root / "trials" / f"trial_{trial:02d}.json"
        folder = root / "trial_fits" / f"trial_{trial:02d}"
        if path.exists():
            row = json.loads(path.read_text())
            if row["run_identity"] != run_id or row["parameters"] != parameters:
                raise ValueError("completed trial identity changed")
            if row["status"] == "finished":
                fit = json.loads((folder / "fit.json").read_text())
                if (
                    digest(folder / "fit.json") != row["fit_sha256"]
                    or digest(folder / "model.eqx") != fit["model_sha256"]
                ):
                    raise ValueError("completed trial fit payload changed")
            trials.append(row)
            continue
        row = {
            "trial": trial,
            "parameters": parameters,
            "seed": seed_base + trial,
            "run_identity": run_id,
            "status": "running",
        }
        started = time.perf_counter()
        try:
            model, fit = fit_once(base, "M2", parameters, "main", row["seed"], batch, folder, run_id)
            member = PhysicsMember(model)
            pred = run_prepared(
                member, replace(data, model_hash=member.state_digest()), cfg, prepared, audit=[]
            )
            row.update(
                status="finished",
                rmse_s=rmse([r.y_true for r in pred], [r.y_pred for r in pred]),
                fit_sha256=digest(folder / "fit.json"),
                fit_recorded_s=fit["elapsed_s"],
            )
        except (CheckpointError, OSError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row.update(
            elapsed_s=time.perf_counter() - started,
            interruption_overhead="Unknown if an uncheckpointed segment was lost",
        )
        write_json(path, row)
        trials.append(row)
        print(json.dumps({"trial": trial, "status": row["status"]}), flush=True)
    successful = [r for r in trials if r["status"] == "finished"]
    if not successful:
        raise RuntimeError("all 20 trial slots failed; no additional search is permitted")
    chosen = min(successful, key=lambda r: (r["rmse_s"], r["trial"]))["parameters"]
    write_json(root / "selection.json", {"parameters": chosen, "rule": method["selection"]})
    return chosen, trials


def run_clean(
    method_path: Path, config: Path, gold_root: Path, root: Path, fold: str, phi_label: str
) -> dict:
    _remote_guard()
    import jax
    import pandas as pd

    from apexpinn.contracts import load_decisions
    from apexpinn.data.gold import build_gold, fit_normalization
    from apexpinn.data.guards import batch_from_gold
    from apexpinn.eval.calibration_report import compute_evaluation, render_report
    from apexpinn.models.ensemble import PhysicsEnsemble, PhysicsMember
    from apexpinn.train.physics_study import fit_once, read_method

    method, decisions = read_spec(method_path), load_decisions(config)
    base_path = Path(method["base_training_method"])
    base = read_method(base_path)
    plan = split_plan(decisions, fold)
    if phi_label not in ("low", "mid", "high") or decisions.status != "frozen":
        raise ValueError("frozen decisions and a declared phi required")
    g0_path = Path("experiments/g0_development_20261103/acceptance.json")
    g1_path = Path("experiments/g1_real_20261103/acceptance.json")
    g0, g1 = json.loads(g0_path.read_text()), json.loads(g1_path.read_text())
    manifest_path = gold_root / f"phi_{phi_label}" / "manifest.json"
    payload = manifest_path.parent / "F1/laps.parquet"
    if (
        digest(config) != g0["instrument"]["current_config_sha256"]
        or g1["conditions"]["G1"] != "PASS"
        or digest(manifest_path) != g0["phi"][phi_label]["gold_manifest_sha256"]
    ):
        raise ValueError("frozen G0/G1/config inputs do not agree")
    expected = json.loads(manifest_path.read_text())["folds"]["F1"]["files"]["laps.parquet"]["sha256"]
    if digest(payload) != expected:
        raise ValueError("Gold input payload changed")
    sources = {
        p.as_posix(): digest(p)
        for d in (Path("src/apexpinn"), Path("scripts"))
        for p in sorted(d.rglob("*.py"))
    }
    run_id = {
        "method_sha256": digest(method_path),
        "base_method_sha256": digest(base_path),
        "config_sha256": digest(config),
        "gold_sha256": expected,
        "source_sha256": sources,
        "G0_sha256": digest(g0_path),
        "G1_sha256": digest(g1_path),
        "plan": plan,
        "phi_label": phi_label,
        "devices": [str(d) + ":" + d.device_kind for d in jax.devices()],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("jax", "jaxlib", "equinox", "optax", "diffrax", "numpy", "pandas")
        },
    }
    declaration = root / "declaration.json"
    if declaration.exists():
        if json.loads(declaration.read_text())["identity"] != run_id:
            raise ValueError("calibration run identity changed; use separate outputs")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("nonempty output has no matching clean-calibration declaration")
        write_json(
            declaration,
            {
                "schema": "apexpinn.calibration_declaration/v1",
                "identity": run_id,
                "method": method,
                "S1": "unchanged",
                "T13_v1": "unchanged",
            },
        )
        for filename in [*sources, str(method_path), str(base_path), str(config)]:
            relative = Path(filename).resolve().relative_to(Path.cwd().resolve())
            target = root / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(filename).read_bytes())
    if (root / "evaluation.json").exists():
        render_report(root)
        return json.loads((root / "evaluation.json").read_text())
    silver = pd.read_parquet(payload)
    tuning = select_rows(silver, plan["tuning_circuits"] + plan["validation_circuits"])
    inner_stats = fit_normalization(
        tuning, decisions, test_fold=fold, training_circuits=plan["tuning_circuits"]
    )
    inner_batch = batch_from_gold(build_gold(tuning, inner_stats, decisions), inner_stats)
    seed_base = (
        method["seed"]
        + 10000 * list(decisions.folds).index(fold)
        + 100 * ("low", "mid", "high").index(phi_label)
    )
    chosen, trials = _run_trials(
        method, base, decisions, tuning, inner_stats, inner_batch, plan, root, run_id, seed_base
    )
    final_frame = select_rows(silver, plan["final_circuits"])
    stats = fit_normalization(
        final_frame, decisions, test_fold=fold, training_circuits=plan["final_circuits"]
    )
    batch = batch_from_gold(build_gold(final_frame, stats, decisions), stats)
    write_json(root / "normalization.json", asdict(stats))
    members, fits = [], []
    for member in range(5):
        fitted, fit = fit_once(
            base,
            "M2",
            chosen,
            "main",
            1_000_000 + seed_base + member,
            batch,
            root / "members" / f"member_{member}",
            run_id,
        )
        members.append(PhysicsMember(fitted))
        fits.append(
            {
                "member": member,
                "fit_sha256": digest(root / "members" / f"member_{member}" / "fit.json"),
                "elapsed_s": fit["elapsed_s"],
                "state_sha256": fit["state_sha256"],
            }
        )
    ensemble = PhysicsEnsemble(tuple(members))
    model_hash = ensemble.state_digest()
    horizons = tuple(method["horizons"])
    stamp = {
        "run_identity": run_id,
        "ensemble_sha256": model_hash,
        "normalization_numeric_sha256": _normalization_id(stats),
    }
    calibration_stats = replace(stats, test_fold=plan["calibration_fold"])
    calibration_frame = select_rows(silver, plan["calibration_circuits"])
    cal_rows = _prediction_cache(
        root / "forecasts/calibration_raw",
        {**stamp, "role": "calibration"},
        lambda: _predict(
            ensemble, calibration_frame, calibration_stats, decisions, fold, "calibration", horizons
        ),
    )
    recipe_path = root / "calibrator.json"
    if recipe_path.exists():
        recipe = json.loads(recipe_path.read_text())
        if (
            recipe["calibration_rows_sha256"] != identity(cal_rows)
            or recipe["plan"] != plan
            or recipe["model_hash"] != model_hash
            or recipe["method_id"] != method["method_id"]
        ):
            raise ValueError("completed calibrator identity changed")
    else:
        calibration_started = time.perf_counter()
        recipe = fit_calibrator(cal_rows, plan, method)
        recipe["elapsed_s"] = time.perf_counter() - calibration_started
        write_json(recipe_path, recipe)
    # Outer labels reach evaluation only after the calibration recipe is fixed and saved.
    test_frame = select_rows(silver, plan["outer_circuits"])
    raw_rows = _prediction_cache(
        root / "forecasts/test_raw",
        {**stamp, "role": "outer_test"},
        lambda: _predict(ensemble, test_frame, stats, decisions, fold, "outer_test", horizons),
    )
    calibrated = _prediction_cache(
        root / "forecasts/test_calibrated",
        {**stamp, "role": "outer_test_calibrated", "recipe_sha256": identity(recipe)},
        lambda: (apply_calibrator(raw_rows, recipe), []),
    )
    if ensemble.state_digest() != model_hash or len({f["state_sha256"] for f in fits}) != 5:
        raise ValueError("ensemble changed during calibration or members are not distinct")
    evaluation = compute_evaluation(raw_rows, calibrated, recipe, plan, trials, fits, run_id)
    essential = [declaration, recipe_path, root / "selection.json", root / "normalization.json"]
    essential += list((root / "trials").glob("*.json"))
    essential += [p for d in (root / "members").iterdir() for p in (d / "fit.json", d / "model.eqx")]
    essential += [p for d in (root / "forecasts").iterdir() for p in d.iterdir() if p.is_file()]
    evaluation["files_sha256"] = {p.relative_to(root).as_posix(): digest(p) for p in essential}
    write_json(root / "evaluation.json", evaluation)
    render_report(root)
    return evaluation
