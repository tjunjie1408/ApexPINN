"""Studio-only setup and status helpers. Uses the image's SDK; scientific code runs in its own uv env."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from apexpinn.cloud.artifacts import digest
from apexpinn.cloud.runtime import require_studio_environment
from apexpinn.cloud.transfer import unpack


def prepare_workspace(spec: dict, infra: dict, home: Path) -> Path:
    context = require_studio_environment()
    if (context["account"], context["region"], context["domain_id"], context["space_name"]) != (
        spec["account_id"],
        spec["region"],
        infra["studio_domain_id"],
        infra["space_name"],
    ):
        raise ValueError("Studio account/region/domain/space differs from the prepared infrastructure")
    if spec["schema"] != "apexpinn.studio_handoff/v1":
        raise ValueError("use the Studio handoff, not the managed-job notebook")
    import boto3

    s3 = boto3.Session(region_name=spec["region"]).client("s3")
    archive = home / "inputs" / spec["bundle_name"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        pending = archive.with_suffix(".download")
        s3.download_file(
            infra["bucket_name"], f"inputs/{spec['bundle_sha256']}/{spec['bundle_name']}", str(pending)
        )
        if digest(pending) != spec["bundle_sha256"]:
            raise ValueError("downloaded S3 input hash mismatch")
        pending.rename(archive)
    if digest(archive) != spec["bundle_sha256"]:
        raise ValueError("existing input ZIP hash mismatch")
    workspace = home / "workspaces" / spec["bundle_sha256"][:16]
    if workspace.exists():
        with zipfile.ZipFile(archive) as zipped:
            manifest = json.loads(zipped.read("bundle_manifest.json"))
        if json.loads((workspace / "bundle_manifest.json").read_text()) != manifest:
            raise ValueError("workspace manifest changed")
        for name, item in manifest["files"].items():
            if digest(workspace / name) != item["sha256"]:
                raise ValueError("workspace payload changed; preserve it and prepare a new execution version")
    else:
        unpack(archive, workspace, expected_sha256=spec["bundle_sha256"], kind="input")
    os.environ["APEXPINN_REMOTE_RUNTIME"] = "studio"
    return workspace


def install_environment(workspace: Path) -> Path:
    require_studio_environment()
    uv = shutil.which("uv")
    if uv is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "uv==0.12.1"], check=True)
        uv = str(Path.home() / ".local/bin/uv")
    subprocess.run(
        [uv, "sync", "--frozen", "--no-dev", "--extra", "cuda", "--python", "3.12"],
        cwd=workspace,
        check=True,
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(workspace / ".venv")},
    )
    python = workspace / ".venv/bin/python"
    subprocess.run(
        [
            str(python),
            "-c",
            "import jax; jax.config.update('jax_enable_x64', True); "
            "from apexpinn.train.cloud_checkpoint import require_remote_gpu; "
            "require_remote_gpu(); print(jax.devices(), jax.config.x64_enabled)",
        ],
        cwd=workspace,
        check=True,
    )
    return python


def progress(root: Path) -> dict:
    records = []
    for path in sorted((root / "run/jobs/M2_F6_mid_main").rglob("progress.json")):
        records.append(
            {
                "fit": str(path.parent.relative_to(root)),
                **json.loads(path.read_text()),
                "file_modified_epoch": path.stat().st_mtime,
            }
        )
    return {
        "job_complete": (root / "run/jobs/M2_F6_mid_main/job.json").is_file(),
        "fits_with_checkpoints": len(records),
        "latest": sorted(records, key=lambda r: r["file_modified_epoch"], reverse=True)[:5],
    }
