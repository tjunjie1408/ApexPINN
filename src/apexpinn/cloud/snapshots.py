"""Verified immutable S3 snapshots shared by Studio and EC2, without model imports."""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from apexpinn.cloud.artifacts import digest
from apexpinn.cloud.transfer import pack, unpack


def require_snapshot_sdk(s3) -> None:
    members = s3.meta.service_model.operation_model("PutObject").input_shape.members
    if not {"IfNoneMatch", "ChecksumSHA256"} <= set(members):
        raise RuntimeError(
            "Update the control environment boto3 SDK before training; snapshot APIs are missing"
        )


def validate_state(root: Path) -> None:
    """Reject a snapshot taken between publication of related checkpoint/model files."""
    for pointer in root.rglob("latest.json"):
        generation = json.loads(pointer.read_text())["generation"]
        if "/" in generation or "\\" in generation or generation.startswith("."):
            raise ValueError("invalid checkpoint generation")
        folder = pointer.parent / generation
        metadata = json.loads((folder / "state.json").read_text())
        if digest(folder / "state.eqx") != metadata["state_sha256"]:
            raise ValueError("checkpoint payload changed while taking snapshot")
    for fit in root.rglob("fit.json"):
        record = json.loads(fit.read_text())
        if record.get("status") == "finished" and digest(fit.parent / "model.eqx") != record["model_sha256"]:
            raise ValueError("finished model payload differs")


def snapshot(
    s3,
    root: Path,
    bucket: str,
    prefix: str,
    identity: dict,
    *,
    identity_file: str = "execution_identity.json",
    schema: str = "apexpinn.remote_snapshot/v1",
) -> dict:
    """Upload a verified immutable ZIP first, then atomically publish its S3 pointer."""
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("symlinks are not valid checkpoint artifacts")
    with tempfile.TemporaryDirectory(dir=root.parent) as temp:
        copied = Path(temp) / "state"
        shutil.copytree(
            root,
            copied,
            symlinks=True,
            ignore=shutil.ignore_patterns(".*", "*.tmp", "bundle_manifest.json"),
        )
        if any(path.is_symlink() for path in copied.rglob("*")):
            raise ValueError("symlinks appeared while taking the checkpoint snapshot")
        if json.loads((copied / identity_file).read_text()) != identity:
            raise ValueError("snapshot execution identity differs")
        validate_state(copied)
        archive = Path(temp) / "checkpoint.zip"
        pack(copied, [p for p in copied.rglob("*") if p.is_file()], archive, kind="results")
        sha = digest(archive)
        key = f"{prefix}/snapshots/{uuid.uuid4().hex}.zip"
        with archive.open("rb") as stream:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                IfNoneMatch="*",
                ServerSideEncryption="AES256",
                ContentType="application/zip",
                ChecksumSHA256=base64.b64encode(bytes.fromhex(sha)).decode(),
                Metadata={"sha256": sha},
            )
        receipt = {
            "schema": schema,
            "key": key,
            "sha256": sha,
            "bytes": archive.stat().st_size,
            "identity": identity,
            "created_at": datetime.now(UTC).isoformat(),
        }
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/latest.json",
            Body=json.dumps(receipt).encode(),
            ServerSideEncryption="AES256",
            ContentType="application/json",
        )
        return receipt


def restore(
    s3,
    root: Path,
    bucket: str,
    prefix: str,
    identity: dict,
    *,
    identity_file: str = "execution_identity.json",
    schema: str = "apexpinn.remote_snapshot/v1",
) -> bool:
    """Restore a compatible verified snapshot only into an absent directory; never roll back local state."""
    if root.exists():
        if json.loads((root / identity_file).read_text()) != identity:
            raise ValueError("local state belongs to a different execution")
        validate_state(root)
        return False
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/latest.json")
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return False
        raise
    receipt = json.loads(obj["Body"].read())
    if receipt.get("schema") != schema or receipt.get("identity") != identity:
        raise ValueError("remote snapshot identity differs")
    if not receipt["key"].startswith(f"{prefix}/snapshots/"):
        raise ValueError("snapshot pointer escaped the run prefix")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as temp:
        archive, restored = Path(temp) / "snapshot.zip", Path(temp) / "restored"
        s3.download_file(bucket, receipt["key"], str(archive))
        unpack(archive, restored, expected_sha256=receipt["sha256"], kind="results")
        if json.loads((restored / identity_file).read_text()) != identity:
            raise ValueError("snapshot payload identity differs")
        validate_state(restored)
        restored.rename(root)
    return True
