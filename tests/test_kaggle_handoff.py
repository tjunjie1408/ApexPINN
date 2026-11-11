"""No fitting, optimizer updates, ODE solves, network calls or real runtime allocation."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from apexpinn.cloud import kaggle_runner as runner
from apexpinn.cloud import runtime, session
from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import pack


def test_local_and_missing_kaggle_markers_are_rejected(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APEXPINN_REMOTE_RUNTIME", "kaggle")
    with pytest.raises(RuntimeError, match="hosted Kaggle"):
        runtime.require_remote_environment()
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    with pytest.raises(RuntimeError, match="hosted Kaggle"):
        runtime.require_kaggle_environment()
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setattr(Path, "is_dir", lambda path: str(path).replace("\\", "/").startswith("/kaggle/"))
    runtime.require_kaggle_environment()
    assert runtime.require_remote_environment() == "kaggle"


def test_one_t4_python312_x64_is_required_without_querying_actual_devices(monkeypatch):
    from apexpinn.train import cloud_checkpoint as checkpoint

    monkeypatch.setenv("APEXPINN_REMOTE_RUNTIME", "kaggle")
    monkeypatch.setattr(runtime, "require_kaggle_environment", lambda: None)
    monkeypatch.setattr(checkpoint.platform, "python_version_tuple", lambda: ("3", "12", "0"))
    fake_jax = SimpleNamespace(
        config=SimpleNamespace(x64_enabled=True),
        default_backend=lambda: "gpu",
        devices=lambda: [SimpleNamespace(device_kind="Tesla T4")],
    )
    monkeypatch.setattr(checkpoint, "jax", fake_jax)
    checkpoint.require_remote_gpu()
    for devices in (
        [SimpleNamespace(device_kind="NVIDIA L4")],
        [SimpleNamespace(device_kind="Tesla T4")] * 2,
    ):
        fake_jax.devices = lambda devices=devices: devices
        with pytest.raises(RuntimeError, match="T4 GPU and float64"):
            checkpoint.require_remote_gpu()


def test_deadline_is_kaggle_only_and_pause_is_not_a_failed_fit(monkeypatch, tmp_path):
    from apexpinn.train import physics_study

    monkeypatch.setenv("APEXPINN_REMOTE_RUNTIME", "colab")
    monkeypatch.setenv("APEXPINN_SESSION_DEADLINE", "1")
    monkeypatch.setattr(session.time, "monotonic", lambda: 2)
    session.check_session()
    monkeypatch.setenv("APEXPINN_REMOTE_RUNTIME", "kaggle")
    with pytest.raises(session.SessionPaused) as error:
        session.check_session()
    assert error.value.code == 75
    monkeypatch.setattr(physics_study, "require_remote_gpu", lambda: None)
    monkeypatch.setattr(physics_study, "check_session", lambda: None)
    monkeypatch.setattr(physics_study, "make_model", lambda *args: object())
    monkeypatch.setattr(physics_study, "tree_digest", lambda tree: "fixture")

    def planned_pause(*args, **kwargs):
        raise session.SessionPaused()

    monkeypatch.setattr(physics_study, "train_remote", planned_pause)
    with pytest.raises(session.SessionPaused):
        physics_study.fit_once(
            {"checkpoint_every": 100}, "M2", {}, "main", 0, object(), tmp_path, {"source": "fixture"}
        )
    assert json.loads((tmp_path / "fit.json").read_text())["status"] == "running"


@pytest.mark.parametrize("outcome", ["PAUSED", "JOB_COMPLETE", "FAILED"])
def test_segment_exports_and_restores_full_saved_state_with_mocked_processes(monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(runner, "require_kaggle_environment", lambda: None)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None))
    workspace, working = tmp_path / "workspace", tmp_path / "working"
    method = workspace / "configs/t13_kaggle_v1.yaml"
    method.parent.mkdir(parents=True)
    method.write_text("mock method only")
    write_json(
        workspace / "bundle_manifest.json",
        {"files": {"configs/t13_kaggle_v1.yaml": {"sha256": digest(method)}}},
    )

    def fake_command(command, *, cwd, logs):
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "attempt.log").write_text("mocked timestamped output")
        if "--self-test" in command:
            oracle = Path(command[command.index("--out") + 1])
            write_json(oracle / "selftest.json", {"results": {"M1": "PASS", "M2": "PASS"}})
            return
        run = Path(command[command.index("--out") + 1])
        fit = run / "jobs/M2_F6_mid_main/trial_fits/trial_17"
        write_json(fit / "fit.json", {"status": "running"})
        write_json(fit / "checkpoints/step_00000100_a/state.json", {"step": 100, "normalizers": {"data": 2}})
        (fit / "checkpoints/step_00000100_a/state.eqx").write_bytes(b"mock optimizer/model/RNG bytes")
        write_json(fit / "checkpoints/latest.json", {"generation": "step_00000100_a"})
        if outcome == "JOB_COMPLETE":
            write_json(run / "jobs/M2_F6_mid_main/job.json", {"schema": "mocked-completed-job"})
        else:
            raise subprocess.CalledProcessError(75 if outcome == "PAUSED" else 1, command)

    monkeypatch.setattr(runner, "run_logged", fake_command)
    args = {"bundle_manifest_sha256": digest(workspace / "bundle_manifest.json")}
    if outcome == "FAILED":
        with pytest.raises(subprocess.CalledProcessError):
            runner.run_segment(Path("python"), workspace, working, **args)
    else:
        result = runner.run_segment(Path("python"), workspace, working, **args)
        assert result["status"] == outcome
    (archive,) = (working / "exports").glob("*.zip")
    root = working / runner.RUN_NAME
    expected = json.loads((root / "execution_identity.json").read_text())
    restored = tmp_path / "new-session"
    runner.restore_run(restored, archive, digest(archive), expected)
    assert (
        restored / "run/jobs/M2_F6_mid_main/trial_fits/trial_17/checkpoints/step_00000100_a/state.eqx"
    ).read_bytes() == b"mock optimizer/model/RNG bytes"
    assert list((restored / "logs").rglob("*.log"))
    with pytest.raises(ValueError, match="changed"):
        runner.restore_run(restored, None, None, {**expected, "method_sha256": "different"})
    assert "APEXPINN_SESSION_DEADLINE" not in __import__("os").environ


def test_restore_rejects_incomplete_values_wrong_hash_and_live_overlay(tmp_path):
    source = tmp_path / "source"
    write_json(source / "execution_identity.json", {"source": "fixed"})
    archive = tmp_path / "resume.zip"
    pack(source, [source / "execution_identity.json"], archive, kind="results")
    with pytest.raises(ValueError, match="together"):
        runner.restore_run(tmp_path / "new", archive, None, {})
    with pytest.raises(ValueError, match="SHA"):
        runner.restore_run(tmp_path / "new", archive, "0" * 64, {})
    with pytest.raises(FileExistsError):
        runner.restore_run(source, archive, digest(archive), {})


def test_planned_pause_is_identified_in_process_monitoring(monkeypatch, tmp_path):
    from apexpinn.cloud import process

    monkeypatch.setenv("APEXPINN_REMOTE_RUNTIME", "kaggle")
    monkeypatch.setattr(process, "require_remote_environment", lambda: "kaggle")

    class FakeProcess:
        pid = 123
        stdout = iter(["checkpoint saved before planned pause\n"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 75

    monkeypatch.setattr(process.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    with pytest.raises(subprocess.CalledProcessError) as error:
        process.run_logged(["mock-training"], cwd=tmp_path, logs=tmp_path / "logs")
    assert error.value.returncode == 75
    record = json.loads(next((tmp_path / "logs").glob("*.json")).read_text())
    assert record["status"] == "paused" and record["ended_at"]
    assert "checkpoint saved" in Path(record["log"]).read_text()


def test_outer_forecast_cache_resumes_without_recomputing_mock_outputs_and_rejects_corruption(tmp_path):
    from test_eval import ToyForecaster, evaluation_fixture

    from apexpinn.eval.prepared import prepare_evaluation, run_prepared

    folds, config = evaluation_fixture()
    data = folds["F1"]
    plan = prepare_evaluation("F1", data, config)
    cache = tmp_path / "forecasts"
    first_audit, second_audit = [], []
    first = run_prepared(ToyForecaster(), data, config, plan, audit=first_audit, checkpoint_dir=cache)

    class MustNotForecast:
        def forecast(self, *args):
            raise AssertionError("completed origin was recomputed")

    resumed = run_prepared(MustNotForecast(), data, config, plan, audit=second_audit, checkpoint_dir=cache)
    assert json.loads(json.dumps([asdict(r) for r in first])) == json.loads(
        json.dumps([asdict(r) for r in resumed])
    )
    assert first_audit == second_audit
    path = next(cache.glob("*.json"))
    row = json.loads(path.read_text())
    row["record"]["y_pred"] += 1
    write_json(path, row)
    with pytest.raises(ValueError, match="payload mismatch"):
        run_prepared(MustNotForecast(), data, config, plan, audit=[], checkpoint_dir=cache)


def test_outer_evaluation_pause_preserves_completed_origins(monkeypatch, tmp_path):
    from test_eval import ToyForecaster, evaluation_fixture

    from apexpinn.eval.prepared import prepare_evaluation, run_prepared

    folds, config = evaluation_fixture()
    data = folds["F1"]
    plan = prepare_evaluation("F1", data, config)
    checks = []

    def pause_after_one():
        checks.append(1)
        if len(checks) == 2:
            raise session.SessionPaused()

    monkeypatch.setattr(session, "check_session", pause_after_one)
    with pytest.raises(session.SessionPaused):
        run_prepared(ToyForecaster(), data, config, plan, audit=[], checkpoint_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 1
    monkeypatch.setattr(session, "check_session", lambda: None)
    calls = []

    class CountingToy(ToyForecaster):
        def forecast(self, *args):
            calls.append(1)
            return super().forecast(*args)

    records = run_prepared(CountingToy(), data, config, plan, audit=[], checkpoint_dir=tmp_path)
    assert len(calls) == len(records) - 1


def test_method_numerical_parity_and_notebook_code_cells_have_safe_defaults():
    sys.path.insert(0, str(Path("scripts").resolve()))
    from prepare_kaggle import notebook

    from apexpinn.train.physics_study import read_method

    original = yaml.safe_load(Path("configs/t13_cloud_v1.yaml").read_text())
    method = read_method(Path("configs/t13_kaggle_v1.yaml"))
    changed = {key for key in original.keys() | method.keys() if original.get(key) != method.get(key)}
    assert changed == {"method_id", "status", "training_location"}
    doc = notebook("a" * 64, "b" * 64)
    assert doc["nbformat"] == 4
    for cell in doc["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell["outputs"] and cell["execution_count"] is None
    code = "\n".join("".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code")
    assert "RUN_TRAINING = False" in code and "START_NEW_RUN = False" in code
    assert "CUDA_VISIBLE_DEVICES='0'" in code
