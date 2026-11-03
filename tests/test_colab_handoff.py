"""Local checks only: no fitting, gradients or optimizer updates are executed by these tests."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from apexpinn.cloud.artifacts import digest
from apexpinn.cloud.transfer import pack, safe_name, unpack
from apexpinn.contracts import PredictionOutput, load_decisions
from apexpinn.models.ensemble import PhysicsEnsemble, PhysicsMember
from apexpinn.models.stints import Scenario, StintContext, StintHistory
from apexpinn.train.cloud_checkpoint import CheckpointError, load_state, save_state, train_remote
from apexpinn.train.physics_study import configurations, read_method


def test_bundle_round_trip_checks_every_payload_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "metric.json").write_text('{"score": 1.25}')
    archive = tmp_path / "results.zip"
    pack(source, [source / "metric.json"], archive, kind="results")
    dest = tmp_path / "imported"
    unpack(archive, dest, expected_sha256=digest(archive), kind="results")
    assert (dest / "metric.json").read_bytes() == (source / "metric.json").read_bytes()
    with pytest.raises(FileExistsError):
        unpack(archive, dest, expected_sha256=digest(archive), kind="results")
    with pytest.raises(ValueError, match="SHA"):
        unpack(archive, tmp_path / "wrong", expected_sha256="0" * 64, kind="results")


@pytest.mark.parametrize(
    "name",
    ["../outside", "/absolute", "a/../b", "a\\b", "D:/x", ".env", "PLAN.md", "a/PROJECT.md", "CON.json"],
)
def test_transport_rejects_paths_and_private_planning_documents(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_malicious_archive_is_not_partially_published(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped.txt", "untrusted")
        z.writestr("bundle_manifest.json", "{}")
    with pytest.raises(ValueError, match="unsafe"):
        unpack(archive, tmp_path / "dest", expected_sha256=digest(archive), kind="results")
    assert not (tmp_path / "escaped.txt").exists() and not (tmp_path / "dest").exists()


def test_full_state_serialization_without_training(tmp_path):
    # Representative parameter, offset, moment and RNG arrays, without creating an optimizer/update.
    state = (
        jnp.array([0.3, 0.5]),
        {"offsets": jnp.array([90.0]), "moment": jnp.array([0.1]), "count": jnp.array(42)},
        jnp.array([0, 7], dtype=jnp.uint32),
    )
    meta = {"identity": {"source": "fixed", "data": "fixed"}, "step": 42, "normalizers": {"data": 12.0}}
    save_state(tmp_path, state, meta)
    restored, actual = load_state(tmp_path, state, meta["identity"])
    import jax

    for left, right in zip(
        jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    assert actual["step"] == 42 and actual["normalizers"] == meta["normalizers"]
    with pytest.raises(CheckpointError):
        load_state(tmp_path, state, {"source": "changed", "data": "fixed"})
    generation = json.loads((tmp_path / "latest.json").read_text())["generation"]
    with (tmp_path / generation / "state.eqx").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(CheckpointError):
        load_state(tmp_path, state, meta["identity"])


def test_training_rejects_local_machine_before_touching_model(tmp_path, monkeypatch):
    monkeypatch.delenv("APEXPINN_REMOTE_RUNTIME", raising=False)
    with pytest.raises(RuntimeError, match="Colab-only"):
        train_remote(object(), object(), None, tmp_path / "never", {})
    assert not (tmp_path / "never").exists()


def test_corrupt_checkpoint_metadata_is_not_a_failed_candidate(tmp_path):
    (tmp_path / "latest.json").write_text("{")
    with pytest.raises(CheckpointError, match="unreadable"):
        load_state(tmp_path, (), {})


def test_no_window_ablation_has_exact_finite_zero_gamma():
    from apexpinn.train.physics_study import make_model

    method = read_method(Path("configs/t13_cloud_v1.yaml"))
    model = make_model(method, "M1", configurations(method, "M1")[0], "no_window", 7)
    np.testing.assert_array_equal(model.physics.gamma, np.zeros(5))
    assert np.isfinite(np.asarray(model.physics.raw_gamma)).all()


class ArrayModel(eqx.Module):
    noise_s: object
    marker: object

    def forecast(self, history, horizon, scenario):
        mean = np.full(horizon, 90.0)
        return PredictionOutput(
            mean, mean[None, :], np.full(horizon, float(self.noise_s)), np.zeros(horizon), np.zeros(horizon)
        )


def test_exact_mixture_does_not_inflate_to_largest_noise():
    history = StintHistory(
        "fixture",
        3,
        np.array([1.0, 2.0, 3.0]),
        np.array([90.0, 90.0, 90.0]),
        np.full(3, 0.5),
        StintContext(2, 0.0, True, 30.0, 0.5),
        3,
    )
    ensemble = PhysicsEnsemble(
        tuple(
            PhysicsMember(ArrayModel(jnp.array(v), jnp.array(i)))
            for i, v in enumerate([1.0, 1.0, 1.0, 1.0, 10.0])
        )
    )
    prediction = ensemble.forecast(history, 1, Scenario(0.5))
    np.testing.assert_allclose(prediction.sigma_s, np.sqrt(20.8))
    assert prediction.sigma_s[0] < 10
    assert prediction.member_sigma_s.shape == (5, 1)


def test_plan_has_declared_budget_and_no_output_side_effect(tmp_path):
    out = tmp_path / "not-created"
    run = subprocess.run(
        [sys.executable, "scripts/run_models.py", "--out", str(out), "--plan"],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(run.stdout)
    assert plan["status"] == "NOT RUN" and plan["tuning_trials"] == 720
    assert plan["calibration_fits"] == plan["final_fits"] == 540
    assert not out.exists()


def test_fixed_grids_and_notebook_are_nonexecuted_artifacts():
    method = read_method(Path("configs/t13_cloud_v1.yaml"))
    assert all(len(configurations(method, model)) == 20 for model in ("M1", "M2"))
    notebook = json.loads(Path("notebooks/colab/train_folds.ipynb").read_text())
    import ast

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_fake_fit_orchestration_counts_trials_calibration_and_final_separately(tmp_path, monkeypatch):
    from apexpinn.eval import physics_report
    from apexpinn.train import physics_study as study
    from apexpinn.train import trainer
    from apexpinn.train.baseline_study import fixture_silver

    def forbidden(*args, **kwargs):
        raise AssertionError("local optimization must never execute")

    monkeypatch.setattr(trainer, "_update", forbidden)
    monkeypatch.setattr(study, "require_colab_gpu", lambda: None)
    attempts = []

    def fake_fit(method, name, parameters, variant, seed, batch, folder, run_id):
        attempts.append(str(folder))
        folder.mkdir(parents=True, exist_ok=True)
        return ArrayModel(jnp.array(0.5), jnp.array(seed)), {"elapsed_s": 0.0}

    monkeypatch.setattr(study, "fit_once", fake_fit)
    monkeypatch.setattr(physics_report, "job_diagnostics", lambda *a: {"representative_stints": []})
    decisions = load_decisions("configs/decisions_v1.yaml")
    method = read_method(Path("configs/t13_cloud_v1.yaml"))
    result = study.run_job(
        fixture_silver(decisions),
        decisions,
        method,
        tmp_path,
        "M1",
        "F1",
        "mid",
        "main",
        {"test": "fake fits only"},
    )
    assert len(attempts) == 30  # 20 candidate fits + 5 fixed calibration fits + 5 final fits.
    assert len(result["trials"]) == 20 and len(result["calibration_fits"]) == len(result["final_fits"]) == 5
    again = study.run_job(
        fixture_silver(decisions),
        decisions,
        method,
        tmp_path,
        "M1",
        "F1",
        "mid",
        "main",
        {"test": "fake fits only"},
    )
    assert again == result and len(attempts) == 30
