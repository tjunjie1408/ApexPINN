"""User-launched EC2 setup/training with full-state S3 backup and persistent attempt logs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.process import run_logged
from apexpinn.cloud.runtime import require_ec2_environment
from apexpinn.cloud.snapshots import require_snapshot_sdk, restore, snapshot
from apexpinn.cloud.transfer import unpack

REMINDER = (
    "After F6: verify the S3 snapshot and results, STOP the EC2 instance, then terminate it and delete "
    "its EBS volume only after backup verification. Browser/terminal closure does not stop billing."
)


def verify_workspace(workspace: Path, archive: Path, expected_sha: str) -> None:
    if digest(archive) != expected_sha:
        raise ValueError("existing EC2 input ZIP changed")
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("bundle_manifest.json"))
    if json.loads((workspace / "bundle_manifest.json").read_text()) != manifest:
        raise ValueError("workspace manifest differs from the verified input")
    for name, item in manifest["files"].items():
        if digest(workspace / name) != item["sha256"]:
            raise ValueError("existing workspace changed; preserve it and use a new execution version")


def client(spec: dict):
    import boto3

    s3 = boto3.Session(region_name=spec["region"]).client("s3")
    require_snapshot_sdk(s3)
    return s3


def check_spec(spec: dict) -> dict:
    context = require_ec2_environment()
    if (
        spec.get("schema") != "apexpinn.ec2_handoff/v1"
        or spec.get("region") != context["region"]
        or spec.get("account_id") != context["account"]
        or spec.get("instance_type") != context["instance_type"]
        or spec.get("selection") != {"model": "M2", "fold": "F6", "phi": "mid", "variant": "main"}
        or spec.get("bucket_name") != "apexpinn-m2-f6-985882743884-ap-southeast-2"
    ):
        raise ValueError("EC2 handoff differs from the approved account, region or F6 selection")
    return context


def prepare(spec: dict, home: Path) -> Path:
    check_spec(spec)
    s3 = client(spec)
    archive = home / spec["bundle_name"]
    home.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        pending = archive.with_suffix(".download")
        s3.download_file(
            spec["bucket_name"], f"inputs/{spec['bundle_sha256']}/{spec['bundle_name']}", str(pending)
        )
        if digest(pending) != spec["bundle_sha256"]:
            raise ValueError("EC2 input download hash mismatch")
        pending.rename(archive)
    if digest(archive) != spec["bundle_sha256"]:
        raise ValueError("existing EC2 input ZIP changed")
    workspace = home / "workspaces" / spec["bundle_sha256"][:16]
    if workspace.exists():
        verify_workspace(workspace, archive, spec["bundle_sha256"])
    else:
        unpack(archive, workspace, expected_sha256=spec["bundle_sha256"], kind="input")
    os.environ["APEXPINN_REMOTE_RUNTIME"] = "ec2"
    subprocess.run(
        ["uv", "sync", "--frozen", "--no-dev", "--extra", "cuda", "--python", "3.12"],
        cwd=workspace,
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(workspace / ".venv")},
        check=True,
    )
    subprocess.run(
        [
            str(workspace / ".venv/bin/python"),
            "-c",
            "import jax; jax.config.update('jax_enable_x64', True); "
            "from apexpinn.train.cloud_checkpoint import require_remote_gpu; "
            "require_remote_gpu(); print(jax.devices(), jax.config.x64_enabled)",
        ],
        cwd=workspace,
        check=True,
    )
    write_json(home / "setup.json", {"workspace": str(workspace), "status": "GPU_CHECK_PASS_NO_FITS"})
    return workspace


def execute(workspace: Path, root: Path) -> None:
    command = [
        str(workspace / ".venv/bin/python"),
        "scripts/run_models.py",
        "--method",
        "configs/t13_ec2_v1.yaml",
    ]
    oracle = root / "restart_oracle"
    source = {
        p.relative_to(workspace).as_posix(): digest(p)
        for folder in ("src/apexpinn", "scripts")
        for p in sorted((workspace / folder).rglob("*.py"))
    }
    receipt = oracle / "selftest.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if (
            record["instrument"]["source_sha256"] != source
            or set(record["results"]) != {"M1", "M2"}
            or not all(value.startswith("PASS:") for value in record["results"].values())
        ):
            raise ValueError("restart oracle source/result mismatch")
    else:
        run_logged(command + ["--out", str(oracle), "--self-test"], cwd=workspace, logs=root / "logs/oracle")
    run_logged(
        command
        + ["--out", str(root / "run"), "--model", "M2", "--fold", "F6", "--phi", "mid", "--variant", "main"],
        cwd=workspace,
        logs=root / "logs/training",
    )


def publish_status(s3, spec: dict, root: Path, status: str, receipt: dict | None) -> None:
    progress = []
    for path in (root / "run/jobs/M2_F6_mid_main").rglob("progress.json"):
        progress.append({"path": path.relative_to(root).as_posix(), **json.loads(path.read_text())})
    value = {
        "observed_at": datetime.now(UTC).isoformat(),
        "process_status": status,
        "latest_snapshot": receipt,
        "progress": progress,
        "job_record_exists": (root / "run/jobs/M2_F6_mid_main/job.json").exists(),
        "scientific_gate": "NOT ACCEPTED BY THIS MONITOR",
    }
    s3.put_object(
        Bucket=spec["bucket_name"],
        Key="ec2-runs/m2-f6-mid-v1/status.json",
        Body=json.dumps(value).encode(),
        ServerSideEncryption="AES256",
        ContentType="application/json",
    )


def run(spec: dict, home: Path) -> None:
    context = check_spec(spec)
    import fcntl

    workspace = home / "workspaces" / spec["bundle_sha256"][:16]
    if not (home / "setup.json").is_file():
        raise RuntimeError("Run EC2 setup first; training does not install dependencies")
    verify_workspace(workspace, home / spec["bundle_name"], spec["bundle_sha256"])
    s3 = client(spec)
    os.environ["APEXPINN_REMOTE_RUNTIME"] = "ec2"
    root = home / "state/m2-f6-mid-v1"
    root.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "apexpinn.ec2_execution/v1",
        "host": context,
        "bundle_sha256": spec["bundle_sha256"],
        "method_sha256": digest(workspace / "configs/t13_ec2_v1.yaml"),
    }
    with (root.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        restore(s3, root, spec["bucket_name"], "ec2-runs/m2-f6-mid-v1", identity)
        root.mkdir(exist_ok=True)
        write_json(root / "execution_identity.json", identity)
        stopped = threading.Event()

        def backup(status: str) -> None:
            receipt = snapshot(s3, root, spec["bucket_name"], "ec2-runs/m2-f6-mid-v1", identity)
            publish_status(s3, spec, root, status, receipt)
            print(json.dumps({"snapshot": receipt}), flush=True)

        def loop() -> None:
            while not stopped.wait(300):
                try:
                    backup("RUNNING")
                except Exception:
                    traceback.print_exc()

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        old = signal.signal(signal.SIGTERM, terminate)
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        failed = False
        try:
            publish_status(s3, spec, root, "STARTING", None)
            execute(workspace, root)
        except BaseException:
            failed = True
            raise
        finally:
            stopped.set()
            thread.join()
            signal.signal(signal.SIGTERM, old)
            print(REMINDER, flush=True)
            try:
                backup("PROCESS_FAILED_OR_INTERRUPTED" if failed else "PROCESS_EXIT_0")
            except Exception:
                if not failed:
                    raise
                traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("setup", "run"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--home", type=Path, default=Path("/opt/apexpinn"))
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.operation == "setup":
        prepare(spec, args.home)
    else:
        run(spec, args.home)


if __name__ == "__main__":
    main()
