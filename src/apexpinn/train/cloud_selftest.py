"""User-executed Colab-only restart oracle. Never invoked by local tests."""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.models.ode_model import ODEModel
from apexpinn.models.pinn import PINN
from apexpinn.models.stints import StintBatch, ctx_features
from apexpinn.train.cloud_checkpoint import require_colab_gpu, train_remote
from apexpinn.train.trainer import TrainConfig


def checkpoint_selftest(out: Path) -> None:
    require_colab_gpu()
    if (out / "selftest.json").exists():
        raise FileExistsError("successful self-test receipt exists; inspect it instead of rerunning")
    sources = {
        str(path).replace("\\", "/"): digest(path)
        for base in (Path("src/apexpinn"), Path("scripts"))
        for path in sorted(base.rglob("*.py"))
    }
    tau = jnp.arange(1, 9, dtype=float)
    batch = StintBatch(
        jnp.array([1, 2]),
        ctx_features(jnp.array([1, 2]), 0.0, True, 30.0, 0.5),
        tau,
        jnp.ones((2, 8), dtype=bool),
        jnp.tile(90 + 0.1 * tau, (2, 1)),
        jnp.full((2, 9), 0.5),
        jnp.array([8, 8]),
    )
    cfg = TrainConfig(n_steps=10, n_warm=2, n_ramp=4, n_colloc=4, log_every=2, float64=True)
    results = {}
    for name in ("M1", "M2"):
        key = jax.random.key(927)
        model = PINN.init(key, train_cfg=cfg) if name == "M1" else ODEModel.init(train_cfg=cfg)
        identity = {"kind": "crafted cloud restart oracle", "model": name, "source_sha256": sources}
        full, full_meta = train_remote(model, batch, key, out / name / "full", identity, checkpoint_every=5)
        progress = out / name / "resumed" / "progress.json"
        cursor = json.loads(progress.read_text())["step"] if progress.exists() else 0
        if cursor < 5:
            train_remote(
                model, batch, key, out / name / "resumed", identity, checkpoint_every=5, stop_after=5
            )
        resumed, resume_meta = train_remote(
            model, batch, key, out / name / "resumed", identity, checkpoint_every=5
        )
        assert full_meta["continuation_state_sha256"] == resume_meta["continuation_state_sha256"]
        for a, b in zip(
            jax.tree_util.tree_leaves(eqx.filter(full, eqx.is_array)),
            jax.tree_util.tree_leaves(eqx.filter(resumed, eqx.is_array)),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
        results[name] = "PASS: model, offsets, optimizer, RNG and continuation state match"
    write_json(
        out / "selftest.json",
        {
            "schema": "apexpinn.cloud_restart_oracle/v1",
            "evidence": "Observed",
            "conditions": {"crafted": True, "not_S1_or_G2": True},
            "instrument": {"source_sha256": sources},
            "results": results,
        },
    )
