"""User-launched Kaggle segments with verified checkpoint handoffs. No Kaggle API credentials."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.process import run_logged
from apexpinn.cloud.runtime import require_kaggle_environment
from apexpinn.cloud.transfer import pack, unpack

RUN_NAME = "apexpinn_m2_f6_kaggle_v1"


def restore_run(root: Path, archive: Path | None, sha256: str | None, expected: dict) -> None:
    """A new session accepts only an explicitly selected, hash-verified prior segment."""
    if (archive is None) != (sha256 is None):
        raise ValueError("resume archive and SHA-256 must be supplied together")
    if archive is not None:
        if root.exists():
            raise FileExistsError("resume only into a new session; do not overlay a live run")
        unpack(archive, root, expected_sha256=sha256, kind="results")
    identity_path = root / "execution_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != expected:
            raise ValueError("Kaggle source/input/method changed; preserve the original execution")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("nonempty run lacks an execution identity")
        write_json(identity_path, expected)


def export_segment(root: Path, exports: Path, segment: str) -> dict:
    """Preserve full-state checkpoints and logs, not just fitted parameters or a report."""
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and not path.name.endswith((".tmp", ".pending"))
        and path.name != "bundle_manifest.json"
    ]
    # Output includes raw state and ZIP. Leave space under Kaggle's documented 20 GB output bound.
    if sum(path.stat().st_size for path in files) > 8 * (1 << 30):
        raise ValueError("state exceeds 8 GiB export budget; preserve raw state and review storage")
    archive = exports / f"f6_{segment}.zip"
    pack(root, files, archive, kind="results")
    receipt = {"archive": str(archive), "sha256": digest(archive), "bytes": archive.stat().st_size}
    write_json(exports / f"f6_{segment}.summary.json", receipt)
    print(json.dumps({"checkpoint_export": receipt}), flush=True)
    return receipt


def run_segment(
    python: Path,
    workspace: Path,
    working: Path,
    *,
    bundle_manifest_sha256: str,
    hours: float = 8,
    resume: Path | None = None,
    resume_sha256: str | None = None,
) -> dict:
    """One cooperative segment; the user publishes its notebook outputs after successful exit."""
    require_kaggle_environment()
    if not 0 < hours <= 8:
        raise ValueError("use a positive segment target of at most 8 hours")
    if digest(workspace / "bundle_manifest.json") != bundle_manifest_sha256:
        raise ValueError("workspace input manifest differs")
    manifest = json.loads((workspace / "bundle_manifest.json").read_text())
    for name, item in manifest["files"].items():
        if digest(workspace / name) != item["sha256"]:
            raise ValueError("workspace payload differs; never resume changed code or data")
    os.environ.update(APEXPINN_REMOTE_RUNTIME="kaggle", CUDA_VISIBLE_DEVICES="0")
    root = working / RUN_NAME
    expected = {
        "schema": "apexpinn.kaggle_execution/v1",
        "input_manifest_sha256": bundle_manifest_sha256,
        "method_sha256": digest(workspace / "configs/t13_kaggle_v1.yaml"),
        "selection": "M2/F6/mid/main",
        "device_policy": "single T4, Python 3.12, float64",
    }
    import fcntl

    working.mkdir(parents=True, exist_ok=True)
    with (working / ".apexpinn_f6.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        restore_run(root, resume, resume_sha256, expected)
        segment = "segment_" + uuid.uuid4().hex[:12]
        status_path = root / "segments" / f"{segment}.json"
        status = {
            "status": "RUNNING",
            "started_at": datetime.now(UTC).isoformat(),
            "target_hours": hours,
            "deadline_policy": "cooperative safe points; not a hard timeout guarantee",
            "interruption_overhead": "Unknown for work not committed to a checkpoint",
        }
        write_json(status_path, status)
        prior = os.environ.get("APEXPINN_SESSION_DEADLINE")
        os.environ["APEXPINN_SESSION_DEADLINE"] = str(time.monotonic() + hours * 3600)
        command = [
            str(python),
            "scripts/run_models.py",
            "--method",
            "configs/t13_kaggle_v1.yaml",
            "--out",
            str(root / "run"),
            "--model",
            "M2",
            "--fold",
            "F6",
            "--phi",
            "mid",
            "--variant",
            "main",
        ]
        try:
            oracle = root / "restart_oracle"
            if not (oracle / "selftest.json").is_file():
                run_logged(
                    [*command[:4], "--out", str(oracle), "--self-test"],
                    cwd=workspace,
                    logs=root / "logs" / "oracle",
                )
            run_logged(command, cwd=workspace, logs=root / "logs" / segment)
            if not (root / "run/jobs/M2_F6_mid_main/job.json").is_file():
                raise RuntimeError("process exited zero without a completed F6 job record")
            status["status"] = "JOB_COMPLETE"
        except subprocess.CalledProcessError as exc:
            status.update(status="PAUSED" if exc.returncode == 75 else "FAILED", returncode=exc.returncode)
            if exc.returncode != 75:
                raise
        except BaseException as exc:
            status.update(status="INTERRUPTED_OR_FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if prior is None:
                os.environ.pop("APEXPINN_SESSION_DEADLINE", None)
            else:
                os.environ["APEXPINN_SESSION_DEADLINE"] = prior
            status["ended_at"] = datetime.now(UTC).isoformat()
            write_json(status_path, status)
            receipt = export_segment(root, working / "exports", segment)
        return {**status, **receipt}
