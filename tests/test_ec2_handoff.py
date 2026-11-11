"""EC2-only guards and transport tests with mocked metadata, SDK and subprocesses. No fitting."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_studio_handoff import Store

from apexpinn.cloud import ec2_runner, runtime, snapshots
from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import pack, unpack

CONTEXT = {
    "account": "985882743884",
    "region": "ap-southeast-2",
    "instance_type": "g6.2xlarge",
    "instance_id": "i-0123456789abcdef0",
    "image_id": "ami-example",
}


def spec():
    return {
        "schema": "apexpinn.ec2_handoff/v1",
        "account_id": CONTEXT["account"],
        "region": CONTEXT["region"],
        "instance_type": "g6.2xlarge",
        "selection": {"model": "M2", "fold": "F6", "phi": "mid", "variant": "main"},
        "bucket_name": "apexpinn-m2-f6-985882743884-ap-southeast-2",
        "bundle_name": "apexpinn_m2_f6_ec2_v1.zip",
        "bundle_sha256": "a" * 64,
    }


def test_ec2_guard_rejects_local_before_contacting_metadata(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *a: pytest.fail("metadata contacted"))
    with pytest.raises(RuntimeError, match="local execution"):
        runtime.require_ec2_environment()


@pytest.mark.parametrize("account,accepted", [("985882743884", True), ("000000000000", False)])
def test_ec2_guard_requires_imdsv2_and_project_identity(monkeypatch, account, accepted):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "is_file", lambda p: True)
    monkeypatch.setattr(Path, "read_text", lambda p: "Amazon EC2")
    calls = []
    document = {
        "accountId": account,
        "region": "ap-southeast-2",
        "instanceType": "g6.2xlarge",
        "architecture": "x86_64",
        "instanceId": "i-0123456789abcdef0",
        "imageId": "ami-example",
    }

    def opened(request, timeout):
        calls.append(request)
        return io.BytesIO(b"token" if request.get_method() == "PUT" else json.dumps(document).encode())

    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=opened))
    if accepted:
        assert runtime.require_ec2_environment() == CONTEXT
    else:
        with pytest.raises(RuntimeError, match="identity differs"):
            runtime.require_ec2_environment()
    assert calls[0].get_method() == "PUT"
    assert calls[1].get_header("X-aws-ec2-metadata-token") == "token"


def test_new_method_preserves_all_scientific_values():
    original = yaml.safe_load(Path("configs/t13_cloud_v1.yaml").read_text())
    method = yaml.safe_load(Path("configs/t13_ec2_v1.yaml").read_text())
    assert {k for k in original.keys() | method.keys() if original.get(k) != method.get(k)} == {
        "method_id",
        "status",
        "training_location",
    }
    from apexpinn.train.physics_study import read_method

    assert read_method(Path("configs/t13_ec2_v1.yaml")) == method


def test_spec_rejects_other_fold_before_sdk(monkeypatch):
    monkeypatch.setattr(ec2_runner, "require_ec2_environment", lambda: CONTEXT)
    value = spec()
    value["selection"]["fold"] = "F1"
    with pytest.raises(ValueError, match="selection"):
        ec2_runner.check_spec(value)


def test_execute_is_mocked_and_selects_only_f6(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ec2_runner, "run_logged", lambda cmd, **kwargs: calls.append(cmd))
    ec2_runner.execute(tmp_path / "workspace", tmp_path / "state")
    assert calls[0][-1] == "--self-test"
    assert calls[1][-8:] == ["--model", "M2", "--fold", "F6", "--phi", "mid", "--variant", "main"]
    assert "configs/t13_ec2_v1.yaml" in calls[0]


def test_generic_ec2_snapshot_verifies_and_never_rolls_back(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    identity = {"source": "same", "host": CONTEXT}
    write_json(root / "execution_identity.json", identity)
    write_json(root / "run/progress.json", {"step": 100})
    s3 = Store()
    receipt = snapshots.snapshot(s3, root, "bucket", "ec2-runs/test", identity)
    assert receipt["schema"] == "apexpinn.remote_snapshot/v1"
    target = tmp_path / "restored"
    assert snapshots.restore(s3, target, "bucket", "ec2-runs/test", identity)
    write_json(target / "run/progress.json", {"step": 200})
    assert not snapshots.restore(s3, target, "bucket", "ec2-runs/test", identity)
    assert json.loads((target / "run/progress.json").read_text())["step"] == 200


def test_workspace_verification_rejects_forged_manifest(tmp_path):
    payload = tmp_path / "source"
    payload.mkdir()
    (payload / "one.py").write_text("value = 1\n")
    archive = tmp_path / "input.zip"
    pack(payload, [payload / "one.py"], archive, kind="input")
    workspace = tmp_path / "workspace"
    unpack(archive, workspace, expected_sha256=digest(archive), kind="input")
    ec2_runner.verify_workspace(workspace, archive, digest(archive))
    (workspace / "one.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="workspace changed"):
        ec2_runner.verify_workspace(workspace, archive, digest(archive))
    manifest = json.loads((workspace / "bundle_manifest.json").read_text())
    manifest["files"]["one.py"]["sha256"] = digest(workspace / "one.py")
    write_json(workspace / "bundle_manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest differs"):
        ec2_runner.verify_workspace(workspace, archive, digest(archive))


@pytest.mark.parametrize(
    "device,x64,backend,accepted",
    [
        ("L4", True, "gpu", True),
        ("T4", True, "gpu", False),
        ("L4", False, "gpu", False),
        ("L4", True, "cpu", False),
    ],
)
def test_ec2_gpu_guard_has_no_fallback(monkeypatch, device, x64, backend, accepted):
    from apexpinn.train import cloud_checkpoint as checkpoint

    monkeypatch.setenv("APEXPINN_REMOTE_RUNTIME", "ec2")
    monkeypatch.setattr(runtime, "require_ec2_environment", lambda: CONTEXT)
    monkeypatch.setattr(checkpoint.platform, "python_version_tuple", lambda: ("3", "12", "0"))
    monkeypatch.setattr(
        checkpoint,
        "jax",
        SimpleNamespace(
            config=SimpleNamespace(x64_enabled=x64),
            default_backend=lambda: backend,
            devices=lambda: [SimpleNamespace(device_kind=device)],
        ),
    )
    if accepted:
        checkpoint.require_remote_gpu()
    else:
        with pytest.raises(RuntimeError, match="no CPU fallback"):
            checkpoint.require_remote_gpu()
