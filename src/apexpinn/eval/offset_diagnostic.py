"""Post-result M1 diagnostics on saved models. No optimizer updates or forecast replacement."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from apexpinn.cloud.artifacts import digest, identity, write_json


def decompose_error(y, shape, mask, offsets) -> dict:
    """Split observed training MSE into within-stint shape error and offset error."""
    y, shape, mask, offsets = (
        np.asarray(y, dtype=float),
        np.asarray(shape, dtype=float),
        np.asarray(mask, dtype=bool),
        np.asarray(offsets, dtype=float),
    )
    if y.ndim != 2 or shape.shape != y.shape or mask.shape != y.shape or offsets.shape != y.shape[:1]:
        raise ValueError("expected matching stint-by-lap arrays and one offset per stint")
    counts = mask.sum(axis=1)
    if not len(counts) or np.any(counts == 0):
        raise ValueError("each stint needs at least one observed lap")
    if not (np.isfinite(y[mask]).all() and np.isfinite(shape[mask]).all() and np.isfinite(offsets).all()):
        raise ValueError("observed data, shapes and offsets must be finite")
    residual = np.where(mask, y - shape, 0.0)
    profiled = residual.sum(axis=1) / counts
    centered = np.where(mask, residual - profiled[:, None], 0.0)
    original = np.where(mask, residual - offsets[:, None], 0.0)
    total = int(counts.sum())
    original_mse, shape_mse = float(np.sum(original**2) / total), float(np.sum(centered**2) / total)
    offset_mse = float(np.sum(counts * (profiled - offsets) ** 2) / total)
    if not np.isclose(original_mse, shape_mse + offset_mse, rtol=1e-10, atol=1e-10):
        raise ValueError("offset decomposition identity failed")
    return {
        "n_stints": len(counts),
        "n_observed_laps": total,
        "trained_offset_rmse_s": float(np.sqrt(original_mse)),
        "profiled_offset_rmse_s": float(np.sqrt(shape_mse)),
        "offset_component_mse_s2": offset_mse,
        "offset_share_of_training_sse": offset_mse / original_mse if original_mse else 0.0,
        "stints": [
            {
                "batch_row": i,
                "n": int(counts[i]),
                "trained_offset_s": float(offsets[i]),
                "profiled_offset_s": float(profiled[i]),
                "shape_rmse_s": float(np.sqrt(np.sum(centered[i] ** 2) / counts[i])),
            }
            for i in range(len(counts))
        ],
    }


def _remote_guard() -> None:
    from apexpinn.train.cloud_checkpoint import require_colab_gpu

    require_colab_gpu()


def _restore_model(folder: Path, batch, method: dict, run_identity: dict):
    """Deserialize the committed full state; initialize an optimizer template but never update it."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from apexpinn.train import trainer as core
    from apexpinn.train.cloud_checkpoint import load_state, tree_digest
    from apexpinn.train.physics_study import make_model

    fit = json.loads((folder / "fit.json").read_text())
    conditions = fit["conditions"]
    if fit["status"] != "finished" or conditions["model"] != "M1" or conditions["variant"] != "main":
        raise ValueError("completed M1 main fit required")
    seed = conditions["seed"]
    template = make_model(method, "M1", conditions["parameters"], "main", seed)
    expected_fit = {
        **run_identity,
        **{k: conditions[k] for k in ("model", "parameters", "variant", "seed")},
        "batch_sha256": tree_digest(batch),
    }
    if fit["identity"] != expected_fit:
        raise ValueError("saved fit source/data identity differs from reconstructed batch")
    params, static = eqx.partition((template, jnp.zeros(batch.n_stints)), eqx.is_inexact_array)
    # Zero moments supply the serialization structure only. There is no optimizer update or gradient.
    state_template = core._optimizer(template.train_cfg.learning_rate, template.train_cfg.grad_clip).init(
        params
    )
    key = jax.random.key(seed)
    collocation_key, _ = jax.random.split(key)
    expected = {
        **expected_fit,
        "config": asdict(template.train_cfg),
        "weights": asdict(template.weights),
        "seed_key": np.asarray(jax.random.key_data(key)).tolist(),
    }
    state, metadata = load_state(
        folder / "checkpoints", (params, state_template, jax.random.key_data(collocation_key)), expected
    )
    if not metadata["step"] == fit["trace"]["step"] == template.train_cfg.n_steps:
        raise ValueError("checkpoint is not the completed fit")
    if tree_digest(state) != fit["trace"]["continuation_state_sha256"]:
        raise ValueError("checkpoint continuation state differs from finished fit")
    model, offsets = eqx.combine(state[0], static)
    if identity(np.asarray(offsets).tolist()) != fit["trace"]["training_offset_sha256"]:
        raise ValueError("checkpoint offsets differ from finished fit")
    if digest(folder / "model.eqx") != fit["model_sha256"]:
        raise ValueError("saved model payload changed")
    saved = eqx.tree_deserialise_leaves(folder / "model.eqx", template)
    with_saved_noise = eqx.tree_at(lambda m: m.noise_s, model, saved.noise_s)
    if tree_digest(with_saved_noise) != fit["state_sha256"]:
        raise ValueError("checkpoint model differs beyond the expected fitted-noise assignment")
    return saved, np.asarray(offsets), fit


def _shape_cases(model, batch, curve_indices: list[int]) -> tuple[np.ndarray, dict, list[dict]]:
    """Evaluate states once on the saved training grid, without gradients or physics solves."""
    import equinox as eqx
    import jax

    @eqx.filter_jit
    def states(m, data):
        return jax.vmap(lambda c, ctx: m.shape(c, ctx, data.tau))(data.compound_idx, data.ctx)

    g, h, w = (np.asarray(v) for v in states(model, batch))
    mask, y, tau = np.asarray(batch.mask), np.asarray(batch.y), np.asarray(batch.tau)
    for name, values in (("shape", g), ("thermal state", h), ("wear state", w)):
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"nonfinite observed {name}")
    stats = {
        "observed_thermal_min": float(h[mask].min()),
        "observed_thermal_max": float(h[mask].max()),
        "observed_wear_min_s": float(w[mask].min()),
        "observed_wear_max_s": float(w[mask].max()),
    }
    curves = []
    for i in curve_indices:
        keep = mask[i]
        curves.append(
            {
                "batch_row": i,
                "compound_idx": int(batch.compound_idx[i]),
                "tau": tau[keep].tolist(),
                "y_s": y[i, keep].tolist(),
                "shape_s": g[i, keep].tolist(),
                "thermal": h[i, keep].tolist(),
                "wear_s": w[i, keep].tolist(),
            }
        )
    return g, stats, curves


def _plot_results(rows: list[dict], output: Path) -> None:
    # Colab exports an inline backend that the isolated project environment cannot import.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = [r for r in rows if r["stage"] == "final"]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    x = np.arange(len(final))
    ax.bar(x - 0.18, [r["trained_offset_rmse_s"] for r in final], 0.36, label="Saved training offsets")
    ax.bar(x + 0.18, [r["profiled_offset_rmse_s"] for r in final], 0.36, label="Closed-form offsets")
    ax.set(
        xticks=x, xlabel="Final member", ylabel="Training RMSE (s)", title="M1/F1/mid: offset contribution"
    )
    ax.legend()
    fig.savefig(output / "offset_errors.png", dpi=150)
    plt.close(fig)
    # Preselected member 0 and middle batch row; no selection using error magnitude.
    example = final[0]["curves"][len(final[0]["curves"]) // 2]
    row = final[0]["stints"][example["batch_row"]]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(example["tau"], example["y_s"], "o", label="Observed training laps", markersize=3)
    for field, label in (("trained_offset_s", "Saved offset"), ("profiled_offset_s", "Closed-form offset")):
        ax.plot(example["tau"], np.asarray(example["shape_s"]) + row[field], label=label)
    ax.set(
        xlabel="Stint lap",
        ylabel="Corrected lap time (s)",
        title=f"Member 0, fixed training batch row {example['batch_row']}",
    )
    ax.legend()
    fig.savefig(output / "example_curve.png", dpi=150)
    plt.close(fig)


def run_diagnostic(run: Path, gold_root: Path, output: Path) -> dict:
    """Read one completed M1 shard and publish a separate post-result diagnostic."""
    _remote_guard()
    import pandas as pd

    from apexpinn.contracts import load_decisions
    from apexpinn.data.gold import build_gold, fit_normalization
    from apexpinn.data.guards import batch_from_gold
    from apexpinn.eval.protocol import inner_circuit_split
    from apexpinn.train.physics_study import read_method

    run, gold_root, output = run.resolve(), gold_root.resolve(), output.resolve()
    if (
        output.exists()
        or output.is_relative_to(run)
        or run.is_relative_to(output)
        or output.is_relative_to(gold_root)
        or gold_root.is_relative_to(output)
    ):
        raise ValueError("choose a new output directory outside the original run and Gold inputs")
    declaration = json.loads((run / "declaration.json").read_text())
    run_id = declaration["identity"]
    # The diagnostic is new code. Every pre-existing execution-source byte must still agree.
    for name, expected in run_id["source_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"execution source changed: {name}")
    for package, expected in run_id["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"package version changed: {package}; use the original locked environment")
    method_path, config_path = run / "source/t13_method.yaml", run / "source/decisions_v1.yaml"
    if digest(method_path) != run_id["method_sha256"] or digest(config_path) != run_id["config_sha256"]:
        raise ValueError("archived method or config changed")
    job_dir = run / "jobs/M1_F1_mid_main"
    job_path = job_dir / "job.json"
    job = json.loads(job_path.read_text())
    if job["run_identity"] != run_id:
        raise ValueError("job declaration mismatch")
    before = {job_path: digest(job_path)}
    for name, expected in job["files"].items():
        path = job_dir / name
        if digest(path) != expected:
            raise ValueError(f"completed job payload changed: {name}")
        before[path] = expected
    payload = gold_root / "phi_mid/F1/laps.parquet"
    if digest(payload) != run_id["gold_sha256"]["mid/F1"]:
        raise ValueError("Gold input differs from completed run")
    before[payload] = run_id["gold_sha256"]["mid/F1"]
    method, decisions = read_method(method_path), load_decisions(config_path)
    gold = pd.read_parquet(payload)
    if decisions.status != "frozen" or not gold.dataset_split.eq("development").all():
        raise ValueError("frozen development data required")
    split = inner_circuit_split(decisions, "F1")
    rows = []
    started = time.perf_counter()
    output.mkdir(parents=True)
    for stage in ("calibration", "final"):
        stats = fit_normalization(
            gold,
            decisions,
            test_fold="F1",
            **({"training_circuits": split.training_circuits} if stage == "calibration" else {}),
        )
        batch = batch_from_gold(build_gold(gold, stats, decisions), stats)
        # Fixed batch rows make plotted cases independent of error magnitude and member selection.
        examples = sorted({0, batch.n_stints // 2, batch.n_stints - 1})
        for member in range(5):
            folder = job_dir / stage / f"member_{member}"
            model, offsets, fit = _restore_model(folder, batch, method, run_id)
            states, state_stats, curves = _shape_cases(model, batch, examples)
            decomposition = decompose_error(batch.y, states, batch.mask, offsets)
            result = {
                "stage": stage,
                "member": member,
                **decomposition,
                **state_stats,
                "curves": curves,
                "fit_sha256": digest(folder / "fit.json"),
                "last_logged_preupdate_mse_s2": fit["trace"]["trace"][-1]["raw"]["data"],
                "quantity": "training-batch residuals and offset decomposition, not forecast scores",
                "population": {"batch_sha256": fit["identity"]["batch_sha256"]},
            }
            write_json(output / f"{stage}_member_{member}.json", result)
            rows.append(result)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "member": member,
                        "trained_rmse_s": result["trained_offset_rmse_s"],
                        "profiled_rmse_s": result["profiled_offset_rmse_s"],
                    }
                ),
                flush=True,
            )
    for path, expected in before.items():
        if digest(path) != expected:
            raise ValueError(f"input changed during diagnosis: {path}")
    receipt = {
        "schema": "apexpinn.m1_offset_diagnostic/v1",
        "evidence": "Observed",
        "quantity": "post-result training-offset and shape diagnostics",
        "population": {"job": "M1_F1_mid_main", "fits": len(rows)},
        "conditions": {
            "method": "m1_offset_diagnostic/v1",
            "optimizer_updates": 0,
            "new_forecasts": False,
            "training": False,
            "G2": "NOT ESTABLISHED",
        },
        "instrument": {
            "source_run": str(run),
            "source_job_sha256": before[job_path],
            "diagnostic_source_sha256": digest(Path(__file__)),
            "gold_sha256": digest(payload),
            "elapsed_s": time.perf_counter() - started,
        },
        "status": "DIAGNOSTIC_COMPLETE",
        "results": rows,
        "limits": (
            "Profiled offsets use all training labels. They are not held-out forecasts or replacement fits."
        ),
    }
    # Commit numerical evidence independently so a renderer failure never requires model replay.
    write_json(output / "computation.json", {**receipt, "status": "DIAGNOSTIC_COMPUTE_COMPLETE"})
    _publish_report(receipt, output)
    return receipt


def _publish_report(receipt: dict, output: Path) -> None:
    """Render saved numbers and curves without importing or evaluating a model."""
    rows = receipt["results"]
    started = time.perf_counter()
    lines = [
        "# M1 training-offset diagnostic",
        "",
        "Observed. Post-result diagnosis only; zero optimizer updates.",
        "Profiled offsets use every scored label in each training stint. "
        "They cannot replace forecast scores.",
        "Source hashes, stage-specific batch identities and exact values: diagnostic.json.",
        "",
        "| Stage | Member | Trained RMSE (s) | Profiled RMSE (s) | Offset share of SSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['member']} | {row['trained_offset_rmse_s']:.4f} | "
            f"{row['profiled_offset_rmse_s']:.4f} | {row['offset_share_of_training_sse']:.2%} |"
        )
    lines += [
        "",
        "A large reduction identifies a training-offset contribution. Residual error after profiling "
        "measures within-stint shape misfit. Neither proves that changing the optimizer will improve "
        "held-out forecasting. No scientific threshold or pivot decision is introduced.",
        "",
    ]
    if receipt["status"] == "RECOVERED_DIAGNOSTIC_COMPLETE":
        lines += [
            "Report recovered from saved per-member JSON after a plotting failure. "
            "The original process exit remains recorded; no models were rerun.",
            "",
        ]
    (output / "diagnostic.md").write_text("\n".join(lines), encoding="utf-8")
    _plot_results(rows, output)
    receipt["instrument"]["render_elapsed_s"] = time.perf_counter() - started
    receipt["files_sha256"] = {
        name: digest(output / name) for name in ("diagnostic.md", "offset_errors.png", "example_curve.png")
    }
    write_json(output / "diagnostic.json", receipt)


def recover_diagnostic(source: Path, run: Path, output: Path, *, process_status: Path | None = None) -> dict:
    """Recover reporting from all ten saved member records; never load models or checkpoints."""
    source, run, output = source.resolve(), run.resolve(), output.resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, run)):
        raise ValueError("choose a new recovery directory outside source inputs")
    job_dir = run / "jobs/M1_F1_mid_main"
    job_path = job_dir / "job.json"
    job = json.loads(job_path.read_text())
    expected_names = {f"{stage}_member_{i}.json" for stage in ("calibration", "final") for i in range(5)}
    if {p.name for p in source.glob("*_member_*.json")} != expected_names:
        raise ValueError("all ten unique calibration/final member JSON files are required")
    inputs = {str(job_path): digest(job_path)}
    rows = []
    for stage in ("calibration", "final"):
        for member in range(5):
            path = source / f"{stage}_member_{member}.json"
            row = json.loads(path.read_text())
            fit_path = job_dir / stage / f"member_{member}" / "fit.json"
            fit_sha = digest(fit_path)
            fit = json.loads(fit_path.read_text())
            if (
                row["stage"] != stage
                or row["member"] != member
                or row["fit_sha256"] != fit_sha
                or fit_sha != job["files"][f"{stage}/member_{member}/fit.json"]
                or row["population"]["batch_sha256"] != fit["identity"]["batch_sha256"]
            ):
                raise ValueError("saved member identity differs from original fit")
            stints = row["stints"]
            if row["n_stints"] != len(stints) or any(s["batch_row"] != i for i, s in enumerate(stints)):
                raise ValueError("saved stint row identities differ")
            n = sum(s["n"] for s in stints)
            if n != row["n_observed_laps"] or any(s["n"] <= 0 for s in stints):
                raise ValueError("saved population counts differ")
            shape_mse = sum(s["n"] * s["shape_rmse_s"] ** 2 for s in stints) / n
            offset_mse = (
                sum(s["n"] * (s["profiled_offset_s"] - s["trained_offset_s"]) ** 2 for s in stints) / n
            )
            comparisons = [
                (shape_mse, row["profiled_offset_rmse_s"] ** 2),
                (offset_mse, row["offset_component_mse_s2"]),
                (shape_mse + offset_mse, row["trained_offset_rmse_s"] ** 2),
                (
                    offset_mse / (shape_mse + offset_mse) if shape_mse + offset_mse else 0.0,
                    row["offset_share_of_training_sse"],
                ),
            ]
            if any(not np.isclose(a, b, rtol=1e-10, atol=1e-10) for a, b in comparisons):
                raise ValueError("saved error decomposition does not reconcile")
            inputs[str(path)], inputs[str(fit_path)] = digest(path), fit_sha
            rows.append(row)
    original_status = json.loads(process_status.read_text()) if process_status else None
    if process_status:
        inputs[str(process_status.resolve())] = digest(process_status)
    receipt = {
        "schema": "apexpinn.m1_offset_diagnostic/v1",
        "evidence": "Observed",
        "quantity": "recovered post-result training-offset and shape diagnostics",
        "population": {"job": "M1_F1_mid_main", "fits": len(rows)},
        "conditions": {
            "method": "m1_offset_diagnostic/v1",
            "training": False,
            "optimizer_updates": 0,
            "model_replay": False,
            "G2": "NOT ESTABLISHED",
        },
        "instrument": {
            "source_run": str(run),
            "source_job_sha256": digest(job_path),
            "recovery_source_sha256": digest(Path(__file__)),
            "inputs_sha256": inputs,
        },
        "original_process": original_status,
        "status": "RECOVERED_DIAGNOSTIC_COMPLETE",
        "results": rows,
        "limits": "Saved diagnostic computations are user-run. Recovery verifies identities and arithmetic, "
        "without repeating model inference. Profiled training errors are not held-out scores.",
    }
    output.mkdir(parents=True)
    for name in sorted(expected_names):
        shutil.copyfile(source / name, output / name)
    _publish_report(receipt, output)
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError("input changed during report recovery")
    return receipt
