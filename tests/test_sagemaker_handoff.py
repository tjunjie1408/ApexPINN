"""No fitting, optimizer updates, ODE solves, Docker builds or AWS calls. All cloud clients are fake."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from apexpinn.cloud import runtime
from apexpinn.cloud import sagemaker_control as control
from apexpinn.cloud import sagemaker_worker as worker
from apexpinn.cloud.artifacts import digest, write_json


class CloudError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self):
        self.objects = {}

    def head_object(self, *, Bucket, Key, **kwargs):
        if Key not in self.objects:
            raise CloudError("404")
        obj = self.objects[Key]
        return {"ContentLength": len(obj["Body"]), "ChecksumSHA256": obj.get("ChecksumSHA256")}

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise CloudError("NoSuchKey")
        obj = self.objects[Key]
        return {"Body": io.BytesIO(obj["Body"]), "ETag": obj["ETag"]}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise CloudError("PreconditionFailed")
        if "IfMatch" in kwargs and self.objects[Key]["ETag"] != kwargs["IfMatch"]:
            raise CloudError("PreconditionFailed")
        body = Body.read() if hasattr(Body, "read") else Body
        self.objects[Key] = {**kwargs, "Body": body, "ETag": hashlib.sha256(body).hexdigest()}
        return {}


class FakeSM:
    def __init__(self):
        self.jobs = {}
        self.creations = 0

    def describe_training_job(self, *, TrainingJobName):
        if TrainingJobName not in self.jobs:
            raise CloudError("ResourceNotFound")
        return self.jobs[TrainingJobName]

    def create_training_job(self, **request):
        self.creations += 1
        self.jobs[request["TrainingJobName"]] = {**request, "TrainingJobStatus": "InProgress"}
        return {"TrainingJobArn": "recorded"}


@pytest.fixture
def launch():
    spec = {
        "schema": "apexpinn.sagemaker_handoff/v1",
        "account_id": "985882743884",
        "region": "ap-southeast-2",
        "bundle_name": "input.zip",
        "bundle_sha256": "a" * 64,
        "max_runtime_seconds": 86400,
        "selection": {"model": "M2", "fold": "F6", "phi": "mid", "variant": "main"},
    }
    infra = {
        "account_id": spec["account_id"],
        "region": spec["region"],
        "bucket_name": "apexpinn-test",
        "execution_role_arn": "arn:aws:iam::985882743884:role/apexpinn-training",
        "repository_url": "985882743884.dkr.ecr.ap-southeast-2.amazonaws.com/apexpinn-m2-f6",
    }
    return spec, infra


def request(launch, *, job="apexpinn-m2-f6-test-a01"):
    spec, infra = launch
    return control.build_request(
        spec,
        infra,
        image_uri=infra["repository_url"] + "@sha256:" + "b" * 64,
        job_name=job,
        run_id="apexpinn-m2-f6-test",
    )


def test_request_fixed_scope_storage_cost_and_network_contract(launch):
    value = request(launch)
    assert value["ResourceConfig"]["InstanceType"] == "ml.g6.2xlarge"
    assert value["ResourceConfig"]["InstanceCount"] == 1
    assert value["StoppingCondition"] == {"MaxRuntimeInSeconds": 86400}
    assert value["CheckpointConfig"]["LocalPath"] == "/opt/ml/checkpoints"
    assert value["EnableManagedSpotTraining"] is False
    assert value["EnableNetworkIsolation"] is True
    assert "KeepAlivePeriodInSeconds" not in value["ResourceConfig"]
    assert value["InputDataConfig"][0]["ChannelName"] == "bundle"


@pytest.mark.parametrize("field,value", [("max_runtime_seconds", 86401), ("max_runtime_seconds", True)])
def test_request_rejects_unbounded_runtime(launch, field, value):
    launch[0][field] = value
    with pytest.raises(ValueError, match="runtime"):
        request(launch)


def test_request_rejects_other_fold_mutable_image_and_foreign_infrastructure(launch):
    spec, infra = launch
    spec["selection"]["fold"] = "F5"
    with pytest.raises(ValueError, match="restricted"):
        request(launch)
    spec["selection"]["fold"] = "F6"
    with pytest.raises(ValueError, match="digest"):
        control.build_request(
            spec,
            infra,
            image_uri=infra["repository_url"] + ":latest",
            job_name="apexpinn-m2-f6-a",
            run_id="apexpinn-m2-f6-a",
        )
    infra["account_id"] = "000000000000"
    with pytest.raises(ValueError, match="identity"):
        request(launch)


def test_conditional_input_upload_verifies_hash_and_never_overwrites(tmp_path):
    archive = tmp_path / "input.zip"
    archive.write_bytes(b"test payload")
    spec = {"bundle_name": archive.name, "bundle_sha256": digest(archive)}
    client = FakeS3()
    uri = control.upload_input(client, "bucket", archive, spec)
    assert control.upload_input(client, "bucket", archive, spec) == uri
    key = next(iter(client.objects))
    client.objects[key]["ChecksumSHA256"] = "corrupt"
    with pytest.raises(ValueError, match="checksum"):
        control.upload_input(client, "bucket", archive, spec)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        control.upload_input(client, "bucket", archive, spec)


def test_duplicate_launch_is_idempotent_and_active_prefix_blocks_second_job(launch):
    sm, s3 = FakeSM(), FakeS3()
    first = request(launch)
    control.submit(sm, s3, first, bucket="bucket")
    assert control.submit(sm, s3, first, bucket="bucket")["TrainingJobStatus"] == "InProgress"
    assert sm.creations == 1
    second = request(launch, job="apexpinn-m2-f6-test-a02")
    with pytest.raises(ValueError, match="reserved"):
        control.submit(sm, s3, second, bucket="bucket", resume=True)


def test_explicit_resume_only_after_failure_with_same_image_input_and_run(launch):
    sm, s3 = FakeSM(), FakeS3()
    first = request(launch)
    control.submit(sm, s3, first, bucket="bucket")
    sm.jobs[first["TrainingJobName"]]["TrainingJobStatus"] = "Failed"
    second = request(launch, job="apexpinn-m2-f6-test-a02")
    with pytest.raises(ValueError, match="reserved"):
        control.submit(sm, s3, second, bucket="bucket")
    changed = {**second, "Environment": {**second["Environment"], "APEXPINN_IMAGE_URI": "different"}}
    with pytest.raises(ValueError, match="identity"):
        control.submit(sm, s3, changed, bucket="bucket", resume=True)
    control.submit(sm, s3, second, bucket="bucket", resume=True)
    assert sm.creations == 2
    sm.jobs[second["TrainingJobName"]]["TrainingJobStatus"] = "Completed"
    with pytest.raises(ValueError, match="reserved"):
        control.submit(sm, s3, request(launch, job="apexpinn-m2-f6-test-a03"), bucket="bucket", resume=True)


def test_submission_failure_leaves_retryable_same_request_reservation(launch, monkeypatch):
    sm, s3 = FakeSM(), FakeS3()
    original = sm.create_training_job
    monkeypatch.setattr(
        sm, "create_training_job", lambda **kwargs: (_ for _ in ()).throw(CloudError("Denied"))
    )
    with pytest.raises(CloudError):
        control.submit(sm, s3, request(launch), bucket="bucket")
    monkeypatch.setattr(sm, "create_training_job", original)
    control.submit(sm, s3, request(launch), bucket="bucket")
    assert sm.creations == 1


def test_local_sagemaker_guard_rejects_before_any_compute(monkeypatch):
    monkeypatch.delenv("TRAINING_JOB_ARN", raising=False)
    with pytest.raises(RuntimeError, match="managed SageMaker"):
        runtime.require_sagemaker_environment()
    with pytest.raises(RuntimeError, match="managed SageMaker"):
        worker.main()


@pytest.mark.parametrize(
    "backend,kind,x64", [("cpu", "CPU", True), ("gpu", "T4", True), ("gpu", "L4", False)]
)
def test_gpu_guard_rejects_cpu_wrong_gpu_or_float32(monkeypatch, backend, kind, x64):
    from apexpinn.train import cloud_checkpoint as module

    monkeypatch.setenv("APEXPINN_REMOTE_RUNTIME", "sagemaker")
    monkeypatch.setattr(runtime, "require_sagemaker_environment", lambda: {})
    monkeypatch.setattr(module.platform, "python_version_tuple", lambda: ("3", "12", "13"))
    monkeypatch.setattr(
        module,
        "jax",
        SimpleNamespace(
            config=SimpleNamespace(x64_enabled=x64),
            default_backend=lambda: backend,
            devices=lambda: [SimpleNamespace(device_kind=kind)],
        ),
    )
    with pytest.raises(RuntimeError, match="L4 GPU and float64"):
        module.require_remote_gpu()


def test_identity_and_image_source_corruption_are_rejected(tmp_path):
    root, image = tmp_path / "state", tmp_path / "image"
    worker.claim_identity(root, {"input": "fixed"})
    worker.claim_identity(root, {"input": "fixed"})
    with pytest.raises(ValueError, match="different"):
        worker.claim_identity(root, {"input": "changed"})
    workspace = tmp_path / "workspace"
    for base in (workspace, image):
        (base / "src").mkdir(parents=True)
        (base / "src/a.py").write_text("value = 1\n")
        (base / "uv.lock").write_text("lock")
        (base / "pyproject.toml").write_text("project")
    worker.verify_sources(workspace, image)
    (image / "src/a.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="source"):
        worker.verify_sources(workspace, image)


def test_worker_selects_only_f6_and_reuses_only_matching_oracle(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "run_logged", lambda command, **kwargs: calls.append(command))
    worker.execute(tmp_path, tmp_path / "root", {"src/a.py": "hash"})
    assert len(calls) == 2 and "--self-test" in calls[0]
    assert calls[1][-8:] == ["--model", "M2", "--fold", "F6", "--phi", "mid", "--variant", "main"]
    write_json(
        tmp_path / "root/restart_oracle/selftest.json",
        {
            "instrument": {"source_sha256": {"src/a.py": "changed"}},
            "results": {"M1": "PASS: mock", "M2": "PASS: mock"},
        },
    )
    with pytest.raises(ValueError, match="source identity"):
        worker.execute(tmp_path, tmp_path / "root", {"src/a.py": "hash"})


def test_heartbeat_failure_records_timestamp_and_propagates_error(tmp_path):
    with pytest.raises(RuntimeError, match="original failure"):
        with worker.record_heartbeat(tmp_path, {"job_name": "mock"}):
            raise RuntimeError("original failure")
    record = json.loads((tmp_path / "worker.json").read_text())
    assert record["status"] == "failed" and record["updated_at"].endswith("+00:00")


def test_new_method_changes_only_execution_metadata():
    original = yaml.safe_load(Path("configs/t13_cloud_v1.yaml").read_text())
    updated = yaml.safe_load(Path("configs/t13_sagemaker_v1.yaml").read_text())
    changed = {key for key in original.keys() | updated.keys() if original.get(key) != updated.get(key)}
    assert changed == {"method_id", "status", "training_location"}
    from apexpinn.train.physics_study import read_method

    assert read_method(Path("configs/t13_sagemaker_v1.yaml")) == updated


def test_sagemaker_context_requires_one_host_and_an_actual_managed_job_marker(monkeypatch):
    monkeypatch.setenv("TRAINING_JOB_ARN", "arn:aws:sagemaker:ap-southeast-2:985882743884:training-job/test")
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.Path, "is_file", lambda self: True)
    monkeypatch.setattr(
        runtime.Path, "read_text", lambda self: json.dumps({"current_host": "algo-1", "hosts": ["algo-1"]})
    )
    assert runtime.require_sagemaker_environment()["account"] == "985882743884"
    monkeypatch.setattr(
        runtime.Path,
        "read_text",
        lambda self: json.dumps({"current_host": "algo-1", "hosts": ["algo-1", "algo-2"]}),
    )
    with pytest.raises(RuntimeError, match="one training host"):
        runtime.require_sagemaker_environment()


def test_notebook_is_unexecuted_python_and_launch_upload_are_disabled():
    from scripts.prepare_sagemaker import notebook

    value = notebook()
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))
    source = "\n".join("".join(c["source"]) for c in value["cells"])
    assert "SUBMIT_TRAINING = False" in source and "UPLOAD_INPUT = False" in source
    assert "drive.mount" not in source and "/content" not in source


def test_prepare_filters_private_docs_and_preserves_base_data(tmp_path, monkeypatch):
    from scripts import prepare_sagemaker as module

    original, repo = tmp_path / "original", tmp_path / "repo"
    original.mkdir()
    retained = [
        *module.EVIDENCE,
        "data/gold_development_frozen_v1/phi_mid/manifest.json",
        "configs/decisions_v1.yaml",
        "AGENTS.md",
        "RECORD.md",
    ]
    for name in retained:
        target = original / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("frozen")
    archive = tmp_path / "original.zip"
    # Recreate a legacy bundle, which predates outbound private-document guards.
    legacy = {"schema": "apexpinn.cloud_bundle/v1", "kind": "input", "files": {}}
    with zipfile.ZipFile(archive, "x") as bundle:
        for name in retained:
            payload = (original / name).read_bytes()
            legacy["files"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            bundle.writestr(name, payload)
        bundle.writestr("bundle_manifest.json", json.dumps(legacy))
    monkeypatch.setattr(module, "BASE_SHA", digest(archive))
    for name in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "src/apexpinn/cloud/sagemaker_control.py",
        "docker/sagemaker/Dockerfile",
        "infra/sagemaker_managed_job/main.tf",
        "infra/sagemaker_managed_job/README.md",
        "infra/sagemaker_managed_job/.gitignore",
        "notebooks/sagemaker/README_managed_jobs.md",
        "notebooks/sagemaker/requirements-control.txt",
    ):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("test")
    (repo / "configs").mkdir()
    for name in ("t13_cloud_v1.yaml", "t13_sagemaker_v1.yaml"):
        (repo / "configs" / name).write_bytes((Path("configs") / name).read_bytes())
    (repo / "configs/decisions_v1.yaml").write_text("frozen")
    output = tmp_path / "handoff"
    spec = module.prepare(archive, output, repo=repo)
    with zipfile.ZipFile(output / spec["bundle_name"]) as bundle:
        assert not ({Path(n).name.lower() for n in bundle.namelist()} & module.PRIVATE)
        assert bundle.read("data/gold_development_frozen_v1/phi_mid/manifest.json") == b"frozen"
    assert digest(output / spec["bundle_name"]) == spec["bundle_sha256"]
    assert digest(output / spec["image_context_name"]) == spec["image_context_sha256"]
    with pytest.raises(FileExistsError):
        module.prepare(archive, output, repo=repo)
