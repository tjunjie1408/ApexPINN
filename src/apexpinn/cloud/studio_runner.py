"""Foreground Studio execution with persistent logs, S3 snapshots and an explicit shutdown reminder."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import traceback
from pathlib import Path

from apexpinn.cloud import snapshots
from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.process import run_logged
from apexpinn.cloud.runtime import require_studio_environment
from apexpinn.cloud.snapshots import require_snapshot_sdk
from apexpinn.cloud.snapshots import validate_state as validate_state
from apexpinn.cloud.transfer import safe_name

REMINDER = (
    "AFTER F6: verify results and S3 backup, then STOP/DELETE the Studio JupyterLab app and verify Deleted. "
    "Closing the browser does not stop GPU billing. Delete the space/EBS only after backup."
)


def snapshot(s3, root: Path, bucket: str, prefix: str, identity: dict) -> dict:
    return snapshots.snapshot(
        s3,
        root,
        bucket,
        prefix,
        identity,
        identity_file="studio_identity.json",
        schema="apexpinn.studio_snapshot/v1",
    )


def restore(s3, root: Path, bucket: str, prefix: str, identity: dict) -> bool:
    return snapshots.restore(
        s3,
        root,
        bucket,
        prefix,
        identity,
        identity_file="studio_identity.json",
        schema="apexpinn.studio_snapshot/v1",
    )


def execute(python: Path, workspace: Path, root: Path) -> None:
    command = [str(python), "scripts/run_models.py", "--method", "configs/t13_studio_v1.yaml"]
    oracle = root / "restart_oracle"
    sources = {
        p.relative_to(workspace).as_posix(): digest(p)
        for folder in ("src/apexpinn", "scripts")
        for p in sorted((workspace / folder).rglob("*.py"))
    }
    receipt = oracle / "selftest.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["instrument"]["source_sha256"] != sources or set(record["results"]) != {"M1", "M2"}:
            raise ValueError("restart oracle source identity differs")
        if not all(v.startswith("PASS:") for v in record["results"].values()):
            raise ValueError("restart oracle did not pass")
    else:
        run_logged(command + ["--out", str(oracle), "--self-test"], cwd=workspace, logs=root / "logs/oracle")
    run_logged(
        command
        + ["--out", str(root / "run"), "--model", "M2", "--fold", "F6", "--phi", "mid", "--variant", "main"],
        cwd=workspace,
        logs=root / "logs/training",
    )


def run(
    python: Path,
    workspace: Path,
    root: Path,
    *,
    bucket: str,
    prefix: str,
    bundle_sha: str,
    backup_seconds: int = 300,
    max_hours: int = 24,
) -> None:
    studio = require_studio_environment()
    if studio["account"] != "985882743884" or studio["region"] != "ap-southeast-2":
        raise ValueError("Studio must be in the declared project account and Sydney")
    if not 30 <= backup_seconds <= 3600 or not 1 <= max_hours <= 24:
        raise ValueError("backup interval or process timeout outside declared bounds")
    if not prefix.startswith("studio-runs/") or safe_name(prefix) != prefix:
        raise ValueError("use a dedicated studio-runs prefix")
    import fcntl

    import boto3

    os.environ["APEXPINN_REMOTE_RUNTIME"] = "studio"
    identity = {
        "schema": "apexpinn.studio_execution/v1",
        "bundle_sha256": bundle_sha,
        "method_sha256": digest(workspace / "configs/t13_studio_v1.yaml"),
        "studio": studio,
    }
    s3 = boto3.Session(region_name=studio["region"]).client("s3")
    require_snapshot_sdk(s3)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + ".launch.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        restore(s3, root, bucket, prefix, identity)
        root.mkdir(exist_ok=True)
        write_json(root / "studio_identity.json", identity)
        stopped = threading.Event()

        def backup_loop() -> None:
            while not stopped.wait(backup_seconds):
                try:
                    receipt = snapshot(s3, root, bucket, prefix, identity)
                    print(json.dumps({"s3_snapshot": receipt}), flush=True)
                except Exception:
                    # EBS retains the current state. Never publish an incomplete S3 snapshot pointer.
                    traceback.print_exc()

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        old_handler = signal.signal(signal.SIGTERM, terminate)
        timer = threading.Timer(max_hours * 3600, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.daemon = True
        thread = threading.Thread(target=backup_loop, daemon=True)
        thread.start()
        timer.start()
        failed = False
        try:
            execute(python, workspace, root)
        except BaseException:
            failed = True
            raise
        finally:
            timer.cancel()
            stopped.set()
            thread.join()
            signal.signal(signal.SIGTERM, old_handler)
            print(REMINDER, flush=True)
            try:
                print(
                    json.dumps({"final_s3_snapshot": snapshot(s3, root, bucket, prefix, identity)}),
                    flush=True,
                )
            except Exception:
                if not failed:
                    raise
                traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--bundle-sha", required=True)
    args = parser.parse_args()
    run(
        args.python,
        args.workspace,
        args.root,
        bucket=args.bucket,
        prefix=args.prefix,
        bundle_sha=args.bundle_sha,
    )


if __name__ == "__main__":
    main()
