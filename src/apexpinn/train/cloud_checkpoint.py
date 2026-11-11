"""Cloud-only continuation of the existing Adam algorithm, with full-state immutable checkpoints."""

from __future__ import annotations

import json
import os
import platform
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from apexpinn.cloud.artifacts import digest, identity, write_json
from apexpinn.cloud.session import check_session, pause_due
from apexpinn.train import trainer as core


class CheckpointError(ValueError):
    """Integrity/source mismatch, not a failed hyperparameter candidate."""


def require_colab_gpu() -> None:
    """Fail before fitting on a local machine or a Colab CPU fallback."""
    if (
        platform.system() != "Linux"
        or os.environ.get("APEXPINN_REMOTE_RUNTIME") != "colab"
        or not Path("/content").is_dir()
    ):
        raise RuntimeError("Training is Colab-only. Open notebooks/colab/train_folds.ipynb on Colab.")
    if not jax.config.x64_enabled or jax.default_backend() != "gpu":
        raise RuntimeError("A remote Colab GPU and jax_enable_x64=True are required; no CPU fallback.")


def require_remote_gpu() -> None:
    """Keep Colab checks and validate explicitly supported remote GPU contexts."""
    runtime = os.environ.get("APEXPINN_REMOTE_RUNTIME")
    if runtime not in {"sagemaker", "studio", "ec2", "kaggle"}:
        require_colab_gpu()
        return
    from apexpinn.cloud.runtime import (
        require_ec2_environment,
        require_kaggle_environment,
        require_sagemaker_environment,
        require_studio_environment,
    )

    if runtime == "kaggle":
        require_kaggle_environment()
    elif runtime == "ec2":
        require_ec2_environment()
    elif runtime == "studio":
        require_studio_environment()
    else:
        require_sagemaker_environment()
    devices = jax.devices()
    if (
        platform.python_version_tuple()[:2] != ("3", "12")
        or not jax.config.x64_enabled
        or jax.default_backend() != "gpu"
        or len(devices) != 1
        or ("T4" if runtime == "kaggle" else "L4") not in devices[0].device_kind
    ):
        device_kind = "T4" if runtime == "kaggle" else "L4"
        raise RuntimeError(
            f"Remote execution requires Python 3.12, one {device_kind} GPU and float64; no CPU fallback"
        )


def save_state(folder: Path, tree: Any, metadata: dict) -> Path:
    """Commit an immutable generation, then update its small pointer after the payload is complete."""
    folder.mkdir(parents=True, exist_ok=True)
    generation = f"step_{metadata['step']:08d}_{uuid.uuid4().hex[:10]}"
    staging = folder / (".pending_" + generation)
    staging.mkdir()
    eqx.tree_serialise_leaves(staging / "state.eqx", tree)
    metadata = {**metadata, "state_sha256": digest(staging / "state.eqx")}
    write_json(staging / "state.json", metadata)
    staging.rename(folder / generation)
    write_json(folder / "latest.json", {"generation": generation})
    return folder / generation


def load_state(folder: Path, template: Any, expected: dict) -> tuple[Any, dict]:
    """Reject a different method/data/source identity or an incomplete/corrupt committed checkpoint."""
    try:
        name = json.loads((folder / "latest.json").read_text())["generation"]
        if not isinstance(name, str) or Path(name).name != name or not name.startswith("step_"):
            raise CheckpointError("unsafe checkpoint pointer")
        path = folder / name
        metadata = json.loads((path / "state.json").read_text())
        if metadata["identity"] != expected or digest(path / "state.eqx") != metadata["state_sha256"]:
            raise CheckpointError("checkpoint identity/payload differs; use a separately versioned run")
        restored = eqx.tree_deserialise_leaves(path / "state.eqx", template)
    except CheckpointError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CheckpointError(
            "committed checkpoint is unreadable or incomplete; preserve and inspect it"
        ) from exc
    return restored, metadata


def tree_digest(tree: Any) -> str:
    """Hash numerical leaves with dtype and shape, excluding pure execution caches."""
    import hashlib

    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array)):
        values = np.asarray(leaf)
        h.update(str((values.shape, values.dtype.str)).encode())
        h.update(values.tobytes())
    return h.hexdigest()


def train_remote(
    model: Any,
    batch: Any,
    key: jax.Array,
    folder: Path,
    run_identity: dict,
    *,
    checkpoint_every: int = 100,
    stop_after: int | None = None,
) -> tuple[Any, dict]:
    """Use the original update/normalization/curriculum functions; no fitting occurs before the guard."""
    require_remote_gpu()
    check_session()
    cfg, weights = model.train_cfg, model.weights
    if checkpoint_every < 1 or weights.state != 0 or batch.h_true is not None:
        raise ValueError("real-data cloud training requires positive checkpoint interval and no true states")
    if not cfg.float64:
        raise ValueError("cloud protocol requires float64")
    started = time.perf_counter()
    model, batch = core._to_float64(model), core._to_float64(batch)
    k_colloc, _ = jax.random.split(key)
    b0 = core._closed_form_offsets(model, batch)
    params, static = eqx.partition((model, b0), eqx.is_inexact_array)
    spec = core._Spec((), bool(model.uses_collocation), cfg.n_colloc, cfg.colloc_extension)
    names = tuple(
        eqx.filter_eval_shape(
            lambda p: core._raw_terms(
                p, static, batch, core._colloc_at(spec, k_colloc, jnp.array(0), batch.n_laps)
            ),
            params,
        )
    )
    spec = replace(spec, active=tuple(n for n in names if getattr(weights, n) > 0))
    optimizer = core._optimizer(cfg.learning_rate, cfg.grad_clip)
    opt_state = optimizer.init(params)
    expected = {
        **run_identity,
        "config": asdict(cfg),
        "weights": asdict(weights),
        "seed_key": np.asarray(jax.random.key_data(key)).tolist(),
    }
    checkpoint = folder / "checkpoints"
    metadata = {"identity": expected, "step": 0, "trace": [], "recorded_elapsed_s": 0.0, "resumed": False}
    if (checkpoint / "latest.json").exists():
        (params, opt_state, key_data), metadata = load_state(
            checkpoint, (params, opt_state, jax.random.key_data(k_colloc)), expected
        )
        k_colloc = jax.random.wrap_key_data(key_data)
        metadata["resumed"] = True
        metadata["uncheckpointed_interruption_overhead"] = "Unknown; not included in recorded elapsed time"
    else:
        raw0, _ = core._evaluate(params, static, batch, k_colloc, jnp.array(0), spec)
        normalizers, floored = core.init_normalizers(
            {n: float(v) for n, v in raw0.items()}, spec.active, cfg.norm_floor, cfg.norm_rel_floor
        )
        metadata.update(normalizers=normalizers, floored_terms=list(floored))
    coefs = {
        n: jnp.asarray(getattr(weights, n) / metadata["normalizers"][n], dtype=b0.dtype) for n in spec.active
    }
    prior_elapsed = metadata["recorded_elapsed_s"]
    end = cfg.n_steps if stop_after is None else min(stop_after, cfg.n_steps)
    if not metadata["step"] <= end <= cfg.n_steps:
        raise ValueError("invalid continuation/stop cursor")
    for step in range(metadata["step"], end):
        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            raw, norms = core._evaluate(params, static, batch, k_colloc, jnp.asarray(step), spec)
            metadata["trace"].append(
                {
                    "step": step,
                    "raw": {n: float(v) for n, v in raw.items()},
                    "normalized_gradient_norm": {
                        n: float(v) / metadata["normalizers"][n] for n, v in norms.items()
                    },
                }
            )
        value, params, opt_state = core._update(
            params,
            static,
            opt_state,
            batch,
            k_colloc,
            jnp.asarray(step),
            core.curriculum_scale(step, cfg.n_warm, cfg.n_ramp),
            coefs,
            spec,
            optimizer,
        )
        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            metadata["trace"][-1]["objective_before_update"] = float(value)
            metadata["trace"][-1]["curriculum"] = float(core.curriculum_scale(step, cfg.n_warm, cfg.n_ramp))
        paused = pause_due()
        if (step + 1) % checkpoint_every == 0 or step + 1 == end or paused:
            if not np.isfinite(float(jax.block_until_ready(value))):
                raise ValueError("nonfinite training objective; this attempt counts as failed")
            metadata.update(step=step + 1, recorded_elapsed_s=prior_elapsed + time.perf_counter() - started)
            save_state(checkpoint, (params, opt_state, jax.random.key_data(k_colloc)), metadata)
            write_json(
                folder / "progress.json",
                {
                    "step": step + 1,
                    "total": cfg.n_steps,
                    "recorded_elapsed_s": metadata["recorded_elapsed_s"],
                },
            )
            print(json.dumps({"fit": str(folder), "step": step + 1, "total": cfg.n_steps}), flush=True)
            if paused:
                check_session()
    fitted, b_train = eqx.combine(params, static)
    if end == cfg.n_steps:
        raw, _ = core._evaluate(params, static, batch, k_colloc, jnp.asarray(end), spec)
        noise = np.sqrt(max(float(raw["data"]), cfg.norm_floor))
        if not np.isfinite(noise):
            raise ValueError("nonfinite final residual")
        fitted = eqx.tree_at(lambda m: m.noise_s, fitted, jnp.asarray(noise))
    metadata.update(
        complete=end == cfg.n_steps,
        recorded_elapsed_s=prior_elapsed + time.perf_counter() - started,
        training_offset_sha256=identity(np.asarray(b_train).tolist()),
        continuation_state_sha256=tree_digest((params, opt_state, jax.random.key_data(k_colloc))),
    )
    return fitted, metadata
