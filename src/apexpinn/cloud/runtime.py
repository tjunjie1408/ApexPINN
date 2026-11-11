"""Environment checks shared by remote execution wrappers, without initializing JAX."""

from __future__ import annotations

import json
import os
import platform
import re
import urllib.request
from pathlib import Path


def require_kaggle_environment() -> None:
    """Require Kaggle's Linux notebook markers before any subprocess or fit."""
    if (
        platform.system() != "Linux"
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or not Path("/kaggle/input").is_dir()
        or not Path("/kaggle/working").is_dir()
    ):
        raise RuntimeError("Use a hosted Kaggle notebook; local execution is forbidden")


def require_ec2_environment() -> dict:
    """Require the approved EC2 host and IMDSv2 identity; never accept a local fallback."""
    vendor = Path("/sys/devices/virtual/dmi/id/sys_vendor")
    if platform.system() != "Linux" or not vendor.is_file() or vendor.read_text().strip() != "Amazon EC2":
        raise RuntimeError("Use the approved EC2 GPU instance; local execution is forbidden")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = "http://169.254.169.254/latest/"
    request = urllib.request.Request(
        endpoint + "api/token", method="PUT", headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}
    )
    with opener.open(request, timeout=3) as response:
        token = response.read().decode()
    request = urllib.request.Request(
        endpoint + "dynamic/instance-identity/document", headers={"X-aws-ec2-metadata-token": token}
    )
    with opener.open(request, timeout=3) as response:
        identity = json.loads(response.read())
    if (
        identity.get("accountId") != "985882743884"
        or identity.get("region") != "ap-southeast-2"
        or identity.get("instanceType") != "g6.2xlarge"
        or identity.get("architecture") != "x86_64"
        or not re.fullmatch(r"i-[a-f0-9]{8,17}", identity.get("instanceId", ""))
    ):
        raise RuntimeError("EC2 identity differs from the approved Sydney account/instance type")
    return {
        "account": identity["accountId"],
        "region": identity["region"],
        "instance_id": identity["instanceId"],
        "instance_type": identity["instanceType"],
        "image_id": identity["imageId"],
    }


def require_sagemaker_environment() -> dict:
    """Require the managed training job context, not a Studio app or a local container."""
    job = os.environ.get("TRAINING_JOB_ARN", "")
    match = re.fullmatch(r"arn:aws:sagemaker:([^:]+):(\d{12}):training-job/([A-Za-z0-9-]+)", job)
    resource = Path("/opt/ml/input/config/resourceconfig.json")
    if platform.system() != "Linux" or not match or not resource.is_file():
        raise RuntimeError("A managed SageMaker training job is required; local execution is forbidden")
    config = json.loads(resource.read_text())
    if config.get("hosts") != [config.get("current_host")] or not config.get("current_host"):
        raise RuntimeError("SageMaker execution requires exactly one training host")
    return {"region": match[1], "account": match[2], "job_name": match[3], "job_arn": job}


def require_studio_environment() -> dict:
    """Require a real Studio JupyterLab app marker, not a training-job impersonation."""
    metadata = Path("/opt/ml/metadata/resource-metadata.json")
    if platform.system() != "Linux" or not metadata.is_file():
        raise RuntimeError("Use a hosted SageMaker Studio JupyterLab app; local execution is forbidden")
    value = json.loads(metadata.read_text())
    match = re.fullmatch(
        r"arn:aws:sagemaker:([^:]+):(\d{12}):app/(d-[A-Za-z0-9]+)/([^/]+)/JupyterLab/([^/]+)",
        value.get("ResourceArn", ""),
    )
    if not match:
        raise RuntimeError("Studio metadata must identify a JupyterLab app")
    return {
        "region": match[1],
        "account": match[2],
        "domain_id": match[3],
        "space_name": match[4],
        "app_name": match[5],
        "app_arn": match[0],
    }


def require_remote_environment() -> str:
    """Fail before launching subprocesses outside the explicitly supported remote contexts."""
    runtime = os.environ.get("APEXPINN_REMOTE_RUNTIME")
    if runtime == "kaggle":
        require_kaggle_environment()
    elif runtime == "ec2":
        require_ec2_environment()
    elif runtime == "sagemaker":
        require_sagemaker_environment()
    elif runtime == "studio":
        require_studio_environment()
    elif runtime != "colab" or not Path("/content").is_dir():
        raise RuntimeError("Use an approved hosted Colab, Kaggle, SageMaker or EC2 environment")
    return runtime
