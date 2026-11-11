"""Managed-job entry point. No fitting is reachable without the explicit SageMaker GPU guard."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.process import run_logged
from apexpinn.cloud.runtime import require_sagemaker_environment
from apexpinn.cloud.transfer import pack, unpack

ROOT = Path("/opt/ml/checkpoints")
WORKSPACE = Path("/opt/ml/workspace")
IMAGE_ROOT = Path("/opt/ml/code")


def verify_sources(workspace: Path, image_root: Path) -> dict:
    """Require the image and S3 bundle to contain the same numerical/execution sources and lockfile."""
    sources = {
        path.relative_to(workspace).as_posix(): digest(path)
        for base in ("src", "scripts")
        for path in sorted((workspace / base).rglob("*.py"))
    }
    image_sources = {
        path.relative_to(image_root).as_posix(): digest(path)
        for base in ("src", "scripts")
        for path in sorted((image_root / base).rglob("*.py"))
    }
    if not sources or sources != image_sources:
        raise ValueError("image and input source trees differ; rebuild the image from this handoff")
    for name in ("uv.lock", "pyproject.toml"):
        if digest(workspace / name) != digest(image_root / name):
            raise ValueError("image dependencies differ from input lockfile")
    return sources


def claim_identity(root: Path, expected: dict) -> None:
    path = root / "worker_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("checkpoint prefix belongs to a different input/image/run")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(
                "nonempty restored checkpoints have no worker identity; inspect before continuing"
            )
        write_json(path, expected)


@contextlib.contextmanager
def record_heartbeat(root: Path, job: dict):
    stopped = threading.Event()
    started = time.monotonic()

    def publish(status: str) -> None:
        event = {
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            "attempt_elapsed_seconds": time.monotonic() - started,
            **job,
        }
        write_json(root / "worker.json", event)
        print(json.dumps({"heartbeat": event}), flush=True)

    def beat() -> None:
        while not stopped.wait(60):
            publish("running")

    publish("running")
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    status = "interrupted"
    try:
        yield
        status = "finished"
    except BaseException:
        status = "failed"
        raise
    finally:
        stopped.set()
        thread.join()
        publish(status)


def execute(workspace: Path, root: Path, sources: dict) -> None:
    """Use the existing CLI for the small restart oracle, then only the requested F6 job."""
    command = [sys.executable, "scripts/run_models.py", "--method", "configs/t13_sagemaker_v1.yaml"]
    oracle = root / "restart_oracle"
    receipt = oracle / "selftest.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["instrument"]["source_sha256"] != sources or set(record["results"]) != {"M1", "M2"}:
            raise ValueError("restart oracle source identity differs")
        if not all(value.startswith("PASS:") for value in record["results"].values()):
            raise ValueError("restart oracle did not pass")
    else:
        run_logged(command + ["--out", str(oracle), "--self-test"], cwd=workspace, logs=root / "logs/oracle")
    run_logged(
        command
        + ["--out", str(root / "run"), "--model", "M2", "--fold", "F6", "--phi", "mid", "--variant", "main"],
        cwd=workspace,
        logs=root / "logs/training",
    )


def export_results(root: Path, output: Path, job_name: str) -> None:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not any(p.startswith(".") for p in path.relative_to(root).parts)
        and path.suffix != ".tmp"
    ]
    if paths:
        pack(root, paths, output / f"{job_name}.zip", kind="results")


def main() -> int:
    job = require_sagemaker_environment()
    if (job["account"], job["region"]) != (os.environ["APEXPINN_ACCOUNT"], os.environ["APEXPINN_REGION"]):
        raise ValueError("managed job account/region differs from declared execution")
    import jax

    jax.config.update("jax_enable_x64", True)
    from apexpinn.train.cloud_checkpoint import require_remote_gpu

    require_remote_gpu()
    archive = Path("/opt/ml/input/data/bundle") / os.environ["APEXPINN_BUNDLE_NAME"]
    unpack(archive, WORKSPACE, expected_sha256=os.environ["APEXPINN_BUNDLE_SHA256"], kind="input")
    sources = verify_sources(WORKSPACE, IMAGE_ROOT)
    claim_identity(
        ROOT,
        {
            "schema": "apexpinn.sagemaker_execution/v1",
            "run_id": os.environ["APEXPINN_RUN_ID"],
            "input_sha256": digest(archive),
            "image_uri": os.environ["APEXPINN_IMAGE_URI"],
            "method_sha256": digest(WORKSPACE / "configs/t13_sagemaker_v1.yaml"),
        },
    )
    os.environ["PYTHONPATH"] = str(WORKSPACE / "src")

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    failed = False
    try:
        with record_heartbeat(ROOT, job):
            execute(WORKSPACE, ROOT, sources)
    except BaseException as exc:
        failed = True
        failure = Path("/opt/ml/output/failure")
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(f"{type(exc).__name__}: {exc}"[:1024], encoding="utf-8")
        traceback.print_exc()
        raise
    finally:
        try:
            export_results(ROOT, Path("/opt/ml/model"), job["job_name"])
        except Exception:
            if not failed:
                raise
            # Preserve the original training failure if a best-effort final ZIP cannot be written.
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
