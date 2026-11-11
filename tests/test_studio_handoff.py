"""Studio transport/setup tests: no AWS calls, notebook execution, model inference or fitting."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from apexpinn.cloud import runtime
from apexpinn.cloud import studio_runner as runner
from apexpinn.cloud import studio_setup as setup
from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.studio_notebook import notebook


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class Store:
    def __init__(self):
        self.objects = {}
        self.fail_upload = False

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if self.fail_upload:
            raise OSError("upload interrupted")
        self.objects[Key] = Body.read() if hasattr(Body, "read") else Body

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise MissingObject()
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])


def test_missing_snapshot_sdk_features_fail_before_training():
    client = SimpleNamespace(
        meta=SimpleNamespace(
            service_model=SimpleNamespace(
                operation_model=lambda name: SimpleNamespace(input_shape=SimpleNamespace(members={}))
            )
        )
    )
    with pytest.raises(RuntimeError, match="before training"):
        runner.require_snapshot_sdk(client)


@pytest.fixture
def state(tmp_path):
    root = tmp_path / "state"
    identity = {"bundle": "unchanged", "studio": "same-app"}
    write_json(root / "studio_identity.json", identity)
    checkpoint = root / "run/trial/checkpoints"
    generation = checkpoint / "step_100"
    generation.mkdir(parents=True)
    (generation / "state.eqx").write_bytes(b"optimizer/RNG/offset test payload, no fitting")
    write_json(generation / "state.json", {"state_sha256": digest(generation / "state.eqx")})
    write_json(checkpoint / "latest.json", {"generation": generation.name})
    return root, identity


def test_snapshot_restore_roundtrip_and_no_local_rollback(state, tmp_path):
    root, identity = state
    s3 = Store()
    receipt = runner.snapshot(s3, root, "bucket", "studio-runs/test", identity)
    assert receipt["key"].startswith("studio-runs/test/snapshots/")
    destination = tmp_path / "restored"
    assert runner.restore(s3, destination, "bucket", "studio-runs/test", identity)
    runner.validate_state(destination)
    (destination / "local_new_progress.txt").write_text("preserve newer local evidence")
    assert not runner.restore(s3, destination, "bucket", "studio-runs/test", identity)
    assert (destination / "local_new_progress.txt").exists()


def test_failed_upload_never_publishes_a_new_pointer(state):
    root, identity = state
    s3 = Store()
    first = runner.snapshot(s3, root, "bucket", "studio-runs/test", identity)
    s3.fail_upload = True
    with pytest.raises(OSError, match="interrupted"):
        runner.snapshot(s3, root, "bucket", "studio-runs/test", identity)
    assert json.loads(s3.objects["studio-runs/test/latest.json"])["sha256"] == first["sha256"]


def test_snapshot_rejects_incomplete_generation_before_upload(state):
    root, identity = state
    (root / "run/trial/checkpoints/step_100/state.eqx").write_bytes(b"changed")
    s3 = Store()
    with pytest.raises(ValueError, match="checkpoint payload"):
        runner.snapshot(s3, root, "bucket", "studio-runs/test", identity)
    assert not s3.objects


def test_restore_rejects_wrong_identity_and_corrupt_archive_without_publishing(state, tmp_path):
    root, identity = state
    s3 = Store()
    receipt = runner.snapshot(s3, root, "bucket", "studio-runs/test", identity)
    destination = tmp_path / "reject"
    with pytest.raises(ValueError, match="identity"):
        runner.restore(s3, destination, "bucket", "studio-runs/test", {"different": "identity"})
    s3.objects[receipt["key"]] += b"corrupted"
    with pytest.raises(ValueError, match="SHA"):
        runner.restore(s3, destination, "bucket", "studio-runs/test", identity)
    assert not destination.exists()


def test_snapshot_rejects_symbolic_links(state, monkeypatch):
    root, identity = state
    (root / "link").write_text("not a real link")
    monkeypatch.setattr(Path, "is_symlink", lambda p: p.name == "link")
    with pytest.raises(ValueError, match="symlinks"):
        runner.snapshot(Store(), root, "bucket", "studio-runs/test", identity)


def test_studio_guard_accepts_only_jupyterlab_metadata(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "is_file", lambda p: True)
    arn = "arn:aws:sagemaker:ap-southeast-2:985882743884:app/d-example/m2-f6/JupyterLab/default"
    monkeypatch.setattr(Path, "read_text", lambda p: json.dumps({"ResourceArn": arn}))
    assert runtime.require_studio_environment()["space_name"] == "m2-f6"
    monkeypatch.setattr(
        Path, "read_text", lambda p: json.dumps({"ResourceArn": arn.replace("JupyterLab", "CodeEditor")})
    )
    with pytest.raises(RuntimeError, match="JupyterLab"):
        runtime.require_studio_environment()


def test_local_studio_run_rejected_before_install_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="local execution"):
        runner.run(
            tmp_path / "python",
            tmp_path,
            tmp_path / "run",
            bucket="bucket",
            prefix="studio-runs/test",
            bundle_sha="a" * 64,
        )
    assert not (tmp_path / "run").exists()


def test_environment_setup_commands_are_mocked_and_use_python312_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "require_studio_environment", lambda: {})
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/fake/uv")
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    python = setup.install_environment(tmp_path)
    assert python == tmp_path / ".venv/bin/python"
    assert calls[0][0] == ["/fake/uv", "sync", "--frozen", "--no-dev", "--extra", "cuda", "--python", "3.12"]
    assert calls[0][1]["env"]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")
    assert "require_remote_gpu" in calls[1][0][-1]


def test_worker_selects_only_f6_and_restart_oracle_without_execution(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_logged", lambda command, **kwargs: calls.append(command))
    runner.execute(tmp_path / "python", tmp_path, tmp_path / "state")
    assert "--self-test" in calls[0]
    assert calls[1][-8:] == ["--model", "M2", "--fold", "F6", "--phi", "mid", "--variant", "main"]
    assert "configs/t13_studio_v1.yaml" in calls[1]


def test_new_studio_method_preserves_numerical_protocol():
    original = yaml.safe_load(Path("configs/t13_cloud_v1.yaml").read_text())
    method = yaml.safe_load(Path("configs/t13_studio_v1.yaml").read_text())
    assert {k for k in original.keys() | method.keys() if original.get(k) != method.get(k)} == {
        "method_id",
        "status",
        "training_location",
    }
    from apexpinn.train.physics_study import read_method

    assert read_method(Path("configs/t13_studio_v1.yaml")) == method


def test_notebook_never_autostarts_or_deletes_compute_and_has_completion_reminder():
    value = notebook()
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    source = "\n".join("".join(c["source"]) for c in value["cells"])
    assert "RUN_TRAINING = False" in source
    assert "DeleteApp" in source and "STOP/DELETE" in source
    assert "delete_app(" not in source and "create_training_job(" not in source


def test_progress_reads_checkpoint_files_without_loading_models(state):
    root, _ = state
    write_json(root / "run/jobs/M2_F6_mid_main/trial_00/progress.json", {"step": 100, "total": 1000})
    status = setup.progress(root)
    assert status["fits_with_checkpoints"] == 1 and not status["job_complete"]
    assert status["latest"][0]["step"] == 100
