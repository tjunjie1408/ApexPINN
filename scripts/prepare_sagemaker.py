"""Prepare a self-contained, hash-verified F6 handoff. File I/O only, never fits or calls AWS."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import pack, unpack

BASE_SHA = "b787682f95e30b303bfe7a41a2739dd40051a200724f079ffcab84a7d5618a03"
PRIVATE = {
    "agents.md",
    "record.md",
    "experimental_blueprint.md",
    "experiment_blueprint.md",
    "decisions_review.md",
    "decision_review.md",
    "project.md",
    "prd.md",
    "plan.md",
    "glossary.md",
}
EVIDENCE = (
    "experiments/g0_development_20261103/acceptance.json",
    "experiments/g1_real_20261103/acceptance.json",
    "experiments/g1_real_20261103/review.json",
    "experiments/pivots.md",
    "reports/s1_identifiability.md",
)


def notebook() -> dict:
    cells = []

    def cell(kind: str, source: str) -> None:
        if kind == "code":
            ast.parse(source)
        item = {
            "id": hashlib.sha256(source.encode()).hexdigest()[:12],
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell(
        "markdown",
        """# ApexPINN M2/F6: SageMaker managed training

Prepared, NOT RUN. This notebook controls a separate L4 training job; it never fits locally.
Use the generated handoff folder. Complete the Terraform and image steps in its README first.
No Studio domain or notebook GPU is required. Closing this notebook does not stop a submitted job.
The separate SageMaker execution declaration preserves the original numerical protocol.
""",
    )
    cell(
        "markdown",
        """## 1. Load the handoff
The control environment needs Python 3.12 and boto3 (`pip install -r requirements-control.txt`).
Use `aws login --profile apexpinn` outside this notebook. Never enter credentials in notebook cells.
""",
    )
    cell(
        "code",
        """import json
import sys
from pathlib import Path

HANDOFF = Path.cwd()  # Folder containing launch_spec.json and control/.
sys.path.insert(0, str(HANDOFF / 'control'))
import sagemaker_control as control

spec = json.loads((HANDOFF / 'launch_spec.json').read_text())
infra = json.loads((HANDOFF / 'infra.json').read_text())
print({'selection': spec['selection'], 'region': spec['region'],
       'compute_status': 'NOT RUN', 'max_hours': spec['max_runtime_seconds'] / 3600})
""",
    )
    cell(
        "markdown",
        """## 2. Verify account and select the immutable image
Set IMAGE_URI to the ECR `repository@sha256:...` printed after the image push. Keep RUN_ID
unchanged when resuming; use a new JOB_NAME attempt only after the prior job is Failed/Stopped.
The default 24-hour runtime bound is a timeout, not a spending cap. Review credits and costs first.
""",
    )
    cell(
        "code",
        """import boto3

PROFILE = 'apexpinn'  # Use None with an existing AWS execution-role session.
IMAGE_URI = 'REPLACE_WITH_ECR_REPOSITORY@sha256:DIGEST'
RUN_ID = 'apexpinn-m2-f6-mid-v1'
JOB_NAME = 'apexpinn-m2-f6-mid-v1-a01'
RESUME = False
session = boto3.Session(profile_name=PROFILE, region_name=spec['region'])
s3, sm = session.client('s3'), session.client('sagemaker')
logs = session.client('logs')
control.verify_account(session.client('sts'), s3, infra,
                       region=spec['region'], account=spec['account_id'])
request = control.build_request(spec, infra, image_uri=IMAGE_URI, job_name=JOB_NAME, run_id=RUN_ID)
print(json.dumps(request, indent=2))
""",
    )
    cell(
        "markdown",
        """## 3. Upload the exact input ZIP
This is a separate, explicit S3 write. It verifies the local archive hash and the uploaded native
S3 SHA-256 checksum, and refuses to replace an existing different object. No GPU starts here.
""",
    )
    cell(
        "code",
        """UPLOAD_INPUT = False
if UPLOAD_INPUT:
    print(control.upload_input(s3, infra['bucket_name'], HANDOFF / spec['bundle_name'], spec))
else:
    print('NOT UPLOADED by this cell. Set UPLOAD_INPUT=True when ready.')
""",
    )
    cell(
        "markdown",
        """## 4. Submit the paid remote job
This is the only launch cell. It runs a small, remote M1/M2 checkpoint restart oracle before
M2/F6/mid/main: 20 tuning trials, five calibration fits and five final fits, then evaluation and
physics diagnostics. F1-F5 are not selected. The oracle is separate validation compute.
No automatic retries, Spot instances, warm pool or inference endpoint are requested.
""",
    )
    cell(
        "code",
        """SUBMIT_TRAINING = False
if SUBMIT_TRAINING:
    print(control.submit(sm, s3, request, bucket=infra['bucket_name'], resume=RESUME))
else:
    print('NOT RUN. Set SUBMIT_TRAINING=True only when you choose to start paid training.')
""",
    )
    cell(
        "markdown",
        """## 5. Monitor without relying on the cell timer
Rerun these read-only cells after reconnecting. AWS reports job state and recorded runtime.
CloudWatch carries stdout/stderr. The S3 heartbeat normally refreshes about every minute plus
SageMaker sync delay; it proves the wrapper is alive, not that optimizer steps are advancing.
Inspect fit progress.json files for actual step movement. Do not relaunch to refresh the display.
""",
    )
    cell(
        "code",
        """print(json.dumps(control.job_status(sm, JOB_NAME), indent=2, default=str))
try:
    print(json.dumps(control.heartbeat(s3, infra['bucket_name'], RUN_ID), indent=2))
except s3.exceptions.NoSuchKey:
    print('No synced heartbeat yet; check the AWS job state and CloudWatch logs.')
""",
    )
    cell(
        "code",
        """LOG_TOKEN = globals().get('LOG_TOKEN')
page = control.recent_logs(logs, JOB_NAME, next_token=LOG_TOKEN)
for event in page.get('events', []):
    print(event['message'], end='' if event['message'].endswith('\\n') else '\\n')
LOG_TOKEN = page.get('nextToken')
""",
    )
    cell(
        "markdown",
        """## 6. Results and explicit recovery
After Completed, ModelArtifacts contains the S3 model.tar.gz URI. It contains the versioned
run ZIP and its receipt. Raw checkpoints, immutable attempt logs, source snapshots and reports
also persist under `runs/RUN_ID/checkpoints/`. Download and verify the ZIP/receipt before scientific
acceptance. G2 and P2 are not established by a single F6 shard.

For Failed/Stopped, inspect the original CloudWatch traceback. To resume compatible state, keep
RUN_ID, bundle hash and image digest unchanged, choose a new JOB_NAME, set RESUME=True and review
the new request. S3 sync can leave incomplete state after an abrupt loss; the runner fails closed
on missing/corrupt committed files. Uncheckpointed lost time is Unknown. Forecast evaluation is
not checkpointed per origin and may replay if interrupted. Never relabel a fresh fit as a resume.

Stopping is separate from monitoring: use AWS Console Stop training job if needed. Completed
training compute is released; S3, ECR and CloudWatch storage continue to incur their own charges.
""",
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (control plane)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def copy_file(source: Path, target: Path) -> None:
    if source.is_symlink() or source.name.lower() in PRIVATE:
        raise ValueError(f"private/link payload forbidden: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def prepare(base: Path, output: Path, *, repo: Path = Path("."), runtime: str = "sagemaker") -> dict:
    if runtime not in {"sagemaker", "studio", "ec2"}:
        raise ValueError("unsupported execution target")
    if output.exists():
        raise FileExistsError("choose a new versioned handoff directory")
    original = yaml.safe_load((repo / "configs/t13_cloud_v1.yaml").read_text())
    method = yaml.safe_load((repo / f"configs/t13_{runtime}_v1.yaml").read_text())
    changed = {key for key in original.keys() | method.keys() if original.get(key) != method.get(key)}
    if changed != {"method_id", "status", "training_location"}:
        raise ValueError("Remote declaration must preserve every numerical/scientific setting")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        tmp = Path(temporary)
        restored, payload, stage = tmp / "base", tmp / "payload", tmp / "handoff"
        manifest = unpack(base, restored, expected_sha256=BASE_SHA, kind="input")
        stage.mkdir()
        retained = [
            name
            for name in manifest["files"]
            if (
                name.startswith(("data/gold_development_frozen_v1/", "data/cloud_baselines/"))
                and Path(name).suffix in {".json", ".parquet"}
            )
        ]
        for name in [*retained, *EVIDENCE]:
            copy_file(restored / name, payload / name)
        for name in ("pyproject.toml", "uv.lock", "README.md"):
            copy_file(repo / name, payload / name)
        for folder, pattern in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml")):
            for source in sorted((repo / folder).rglob(pattern)):
                copy_file(source, payload / source.relative_to(repo))
        if (
            digest(payload / "configs/decisions_v1.yaml")
            != manifest["files"]["configs/decisions_v1.yaml"]["sha256"]
        ):
            raise ValueError("frozen decisions differ from verified base input")
        write_json(
            payload / "handoff.json",
            {
                "schema": f"apexpinn.{runtime}_input/v1",
                "base_sha256": BASE_SHA,
                "numerical_protocol": "unchanged from t13_cloud_v1.yaml",
                "compute_status": "NOT RUN",
                "selection": "M2/F6/mid/main",
            },
        )
        archive = stage / f"apexpinn_m2_f6_{runtime}_v1.zip"
        files = [path for path in payload.rglob("*") if path.is_file()]
        pack(payload, files, archive, kind="input")
        spec = {
            "schema": f"apexpinn.{runtime}_handoff/v1",
            "region": "ap-southeast-2",
            "account_id": "985882743884",
            "selection": {"model": "M2", "fold": "F6", "phi": "mid", "variant": "main"},
            "bundle_name": archive.name,
            "bundle_sha256": digest(archive),
            "max_runtime_seconds": 86400,
            "compute_status": "NOT RUN",
            "instance_type": "g6.2xlarge" if runtime == "ec2" else "ml.g6.2xlarge",
        }
        if runtime == "ec2":
            spec["bucket_name"] = "apexpinn-m2-f6-985882743884-ap-southeast-2"
        if runtime == "sagemaker":
            for source in files:
                relative = source.relative_to(payload)
                if relative.parts[0] in {"src", "scripts"} or relative.as_posix() in {
                    "pyproject.toml",
                    "uv.lock",
                    "README.md",
                }:
                    copy_file(source, stage / "image" / relative)
            copy_file(repo / "docker/sagemaker/Dockerfile", stage / "image/Dockerfile")
            build_archive = stage / "image-context.zip"
            with zipfile.ZipFile(build_archive, "x", compression=zipfile.ZIP_DEFLATED) as image_zip:
                for path in sorted((stage / "image").rglob("*")):
                    if path.is_file():
                        image_zip.write(path, path.relative_to(stage / "image").as_posix())
            spec.update(image_context_name=build_archive.name, image_context_sha256=digest(build_archive))
            copy_file(
                repo / "src/apexpinn/cloud/sagemaker_control.py", stage / "control/sagemaker_control.py"
            )
            template = notebook()
            infra_dir = repo / "infra/sagemaker_managed_job"
            readme = repo / "notebooks/sagemaker/README_managed_jobs.md"
        elif runtime == "studio":
            from apexpinn.cloud.studio_notebook import notebook as studio_notebook

            for name in (
                "__init__.py",
                "cloud/__init__.py",
                "cloud/artifacts.py",
                "cloud/transfer.py",
                "cloud/runtime.py",
                "cloud/process.py",
                "cloud/snapshots.py",
                "cloud/studio_setup.py",
                "cloud/studio_runner.py",
            ):
                copy_file(repo / "src/apexpinn" / name, stage / "control/apexpinn" / name)
            template = studio_notebook()
            infra_dir = repo / "infra/sagemaker"
            readme = repo / "notebooks/sagemaker/README.md"
        else:
            for name in (
                "__init__.py",
                "cloud/__init__.py",
                "cloud/artifacts.py",
                "cloud/transfer.py",
                "cloud/runtime.py",
                "cloud/process.py",
                "cloud/snapshots.py",
                "cloud/ec2_runner.py",
            ):
                copy_file(repo / "src/apexpinn" / name, stage / "control/apexpinn" / name)
            template = None
            infra_dir = repo / "infra/ec2"
            readme = repo / "docs/ec2_m2_f6.md"
        write_json(stage / "launch_spec.json", spec)
        for source in sorted(infra_dir.glob("*.tf")):
            copy_file(source, stage / "terraform" / source.name)
        copy_file(infra_dir / "README.md", stage / "terraform/README.md")
        copy_file(infra_dir / ".gitignore", stage / "terraform/.gitignore")
        provider_lock = infra_dir / ".terraform.lock.hcl"
        if provider_lock.exists():
            copy_file(provider_lock, stage / "terraform/.terraform.lock.hcl")
        copy_file(readme, stage / "README.md")
        requirements = (
            repo / "infra/ec2/requirements-control.txt"
            if runtime == "ec2"
            else repo / "notebooks/sagemaker/requirements-control.txt"
        )
        copy_file(requirements, stage / "requirements-control.txt")
        if runtime == "ec2":
            copy_file(infra_dir / "bootstrap.sh.tftpl", stage / "terraform/bootstrap.sh.tftpl")
            write_json(
                stage / "start_training.json",
                {
                    "commands": [
                        "systemctl start apexpinn-train.service",
                        "systemctl show apexpinn-train.service --property=ActiveState,SubState",
                    ]
                },
            )
        else:
            write_json(stage / "train_m2_f6.ipynb", template)
        if runtime == "ec2":
            control_archive = stage / "apexpinn_ec2_control_v1.zip"
            control_files = list((stage / "control").rglob("*.py")) + [
                stage / "launch_spec.json",
                stage / "requirements-control.txt",
            ]
            pack(stage, control_files, control_archive, kind="input")
            control_sha = digest(control_archive)
            write_json(
                stage / "control_receipt.json",
                {
                    "name": control_archive.name,
                    "sha256": control_sha,
                    "key": f"inputs/{control_sha}/{control_archive.name}",
                    "bytes": control_archive.stat().st_size,
                    "input_sha256": spec["bundle_sha256"],
                },
            )
        write_json(
            stage / "handoff_manifest.json",
            {
                "schema": f"apexpinn.{runtime}_package/v1",
                "compute_status": "NOT RUN",
                "base_sha256": BASE_SHA,
                "bundle_sha256": spec["bundle_sha256"],
                "files": {
                    path.relative_to(stage).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
                    for path in sorted(stage.rglob("*"))
                    if path.is_file()
                },
            },
        )
        stage.rename(output)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/colab_exports/apexpinn_t13_input_v1.zip"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", choices=("studio", "sagemaker", "ec2"), default="studio")
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.out, runtime=args.runtime), indent=2))


if __name__ == "__main__":
    main()
