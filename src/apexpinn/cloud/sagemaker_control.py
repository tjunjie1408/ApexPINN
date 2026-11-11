"""Small SageMaker control plane, distributable without JAX or training dependencies."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

TERMINAL = {"Completed", "Failed", "Stopped"}


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _missing(exc: Exception) -> bool:
    return getattr(exc, "response", {}).get("Error", {}).get("Code") in {
        "404",
        "NoSuchKey",
        "NotFound",
        "ResourceNotFound",
        "ResourceNotFoundException",
    }


def verify_account(sts, s3, infra: dict, *, region: str, account: str) -> None:
    if sts.get_caller_identity()["Account"] != account:
        raise ValueError("AWS account differs from the declared project")
    actual = s3.get_bucket_location(Bucket=infra["bucket_name"])["LocationConstraint"] or "us-east-1"
    if actual != region:
        raise ValueError("S3 bucket must be in the declared Sydney region")
    if infra["region"] != region or infra["account_id"] != account:
        raise ValueError("Terraform outputs belong to a different region/account")


def upload_input(s3, bucket: str, archive: Path, spec: dict) -> str:
    """Upload only the exact prepared archive, with conditional creation and native checksum verification."""
    sha = file_sha(archive)
    if sha != spec["bundle_sha256"] or archive.name != spec["bundle_name"]:
        raise ValueError("input ZIP differs from the prepared handoff")
    key = f"inputs/{sha}/{archive.name}"
    checksum = base64.b64encode(bytes.fromhex(sha)).decode()
    try:
        head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except Exception as exc:
        if not _missing(exc):
            raise
        with archive.open("rb") as stream:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                ContentType="application/zip",
                Metadata={"sha256": sha},
                ChecksumSHA256=checksum,
                ServerSideEncryption="AES256",
                IfNoneMatch="*",
            )
        head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    if head.get("ChecksumSHA256") != checksum or head["ContentLength"] != archive.stat().st_size:
        raise ValueError("S3 input checksum/size verification failed")
    return f"s3://{bucket}/{key}"


def build_request(spec: dict, infra: dict, *, image_uri: str, job_name: str, run_id: str) -> dict:
    """Construct one fixed F6 job. Pure preparation, no AWS calls."""
    if spec.get("schema") != "apexpinn.sagemaker_handoff/v1" or spec.get("selection") != {
        "model": "M2",
        "fold": "F6",
        "phi": "mid",
        "variant": "main",
    }:
        raise ValueError("this launcher is restricted to M2/F6/mid/main")
    region, account = spec["region"], spec["account_id"]
    if (region, account) != (infra["region"], infra["account_id"]):
        raise ValueError("infrastructure identity differs from the handoff")
    if not re.fullmatch(r"[a-f0-9]{64}", spec["bundle_sha256"]):
        raise ValueError("a full input SHA-256 is required")
    repository = infra["repository_url"]
    if not re.fullmatch(re.escape(repository) + r"@sha256:[a-f0-9]{64}", image_uri):
        raise ValueError("pin an image digest from the Terraform ECR repository, not a mutable tag")
    if not repository.startswith(f"{account}.dkr.ecr.{region}.amazonaws.com/"):
        raise ValueError("ECR image must belong to this project account and region")
    for value in (job_name, run_id):
        if not re.fullmatch(r"apexpinn-m2-f6-[a-z0-9-]{1,45}", value) or len(value) > 63:
            raise ValueError("job/run names must start apexpinn-m2-f6- and contain lowercase letters/digits")
    seconds = spec["max_runtime_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 3600 <= seconds <= 86400:
        raise ValueError("runtime must be between one and24 hours")
    role = infra["execution_role_arn"]
    if not re.fullmatch(rf"arn:aws:iam::{account}:role/[A-Za-z0-9+=,.@_/-]+", role):
        raise ValueError("execution role must belong to the declared account")
    bucket = infra["bucket_name"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise ValueError("invalid S3 bucket name")
    prefix = f"s3://{bucket}/runs/{run_id}"
    return {
        "TrainingJobName": job_name,
        "RoleArn": role,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["/opt/venv/bin/python", "-m", "apexpinn.cloud.sagemaker_worker"],
        },
        "InputDataConfig": [
            {
                "ChannelName": "bundle",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/inputs/{spec['bundle_sha256']}/{spec['bundle_name']}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/zip",
                "CompressionType": "None",
                "InputMode": "File",
            }
        ],
        "OutputDataConfig": {"S3OutputPath": f"{prefix}/artifacts"},
        "CheckpointConfig": {"S3Uri": f"{prefix}/checkpoints", "LocalPath": "/opt/ml/checkpoints"},
        "ResourceConfig": {"InstanceType": "ml.g6.2xlarge", "InstanceCount": 1, "VolumeSizeInGB": 50},
        "StoppingCondition": {"MaxRuntimeInSeconds": seconds},
        "EnableManagedSpotTraining": False,
        "EnableNetworkIsolation": True,
        "EnableInterContainerTrafficEncryption": True,
        "ProfilerConfig": {"DisableProfiler": True},
        "Environment": {
            "APEXPINN_REMOTE_RUNTIME": "sagemaker",
            "APEXPINN_BUNDLE_SHA256": spec["bundle_sha256"],
            "APEXPINN_BUNDLE_NAME": spec["bundle_name"],
            "APEXPINN_IMAGE_URI": image_uri,
            "APEXPINN_RUN_ID": run_id,
            "APEXPINN_ACCOUNT": account,
            "APEXPINN_REGION": region,
        },
        "Tags": [{"Key": "Project", "Value": "ApexPINN"}, {"Key": "RunId", "Value": run_id}],
    }


def _describe(sm, name: str) -> dict | None:
    try:
        return sm.describe_training_job(TrainingJobName=name)
    except Exception as exc:
        if _missing(exc):
            return None
        # Missing jobs can be ValidationException; unrelated validation failures must still propagate.
        error = getattr(exc, "response", {}).get("Error", {})
        if error.get("Code") == "ValidationException" and "Could not find" in error.get("Message", ""):
            return None
        raise


def submit(sm, s3, request: dict, *, bucket: str, resume: bool = False) -> dict:
    """Reserve one checkpoint prefix with S3 conditional writes before an explicit job submission."""
    run_id = request["Environment"]["APEXPINN_RUN_ID"]
    key = f"locks/{run_id}.json"
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    lease = {"job_name": request["TrainingJobName"], "request_sha256": fingerprint}
    condition = {"IfNoneMatch": "*"}
    try:
        existing = s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if not _missing(exc):
            raise
    else:
        previous = json.loads(existing["Body"].read())
        old_job = _describe(sm, previous["job_name"])
        if previous == lease:
            if old_job:
                return old_job
            condition = {"IfMatch": existing["ETag"]}
        else:
            if not resume or not old_job or old_job["TrainingJobStatus"] not in {"Failed", "Stopped"}:
                raise ValueError("checkpoint prefix is reserved; resume only a terminal Failed/Stopped job")
            previous_env = old_job.get("Environment", {})
            for name in ("APEXPINN_IMAGE_URI", "APEXPINN_BUNDLE_SHA256", "APEXPINN_RUN_ID"):
                if previous_env.get(name) != request["Environment"][name]:
                    raise ValueError("resume image/input/run identity differs")
            condition = {"IfMatch": existing["ETag"]}
    if _describe(sm, request["TrainingJobName"]):
        raise ValueError("job name already exists outside the matching submission receipt")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(lease).encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
        **condition,
    )
    return sm.create_training_job(**request)


def job_status(sm, name: str) -> dict:
    job = sm.describe_training_job(TrainingJobName=name)
    keys = (
        "TrainingJobName",
        "TrainingJobStatus",
        "SecondaryStatus",
        "FailureReason",
        "CreationTime",
        "TrainingStartTime",
        "TrainingEndTime",
        "TrainingTimeInSeconds",
        "BillableTimeInSeconds",
        "ModelArtifacts",
    )
    return {key: job[key] for key in keys if key in job}


def recent_logs(logs, name: str, *, next_token: str | None = None) -> dict:
    params = {"logGroupName": "/aws/sagemaker/TrainingJobs", "logStreamNamePrefix": name + "/", "limit": 50}
    if next_token:
        params["nextToken"] = next_token
    return logs.filter_log_events(**params)


def heartbeat(s3, bucket: str, run_id: str) -> dict:
    response = s3.get_object(Bucket=bucket, Key=f"runs/{run_id}/checkpoints/worker.json")
    return json.loads(response["Body"].read())
