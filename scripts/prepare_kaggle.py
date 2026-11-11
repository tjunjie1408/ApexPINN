"""Build the private Kaggle F6 input and notebook. File I/O only; no API calls or fitting."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tempfile
from pathlib import Path

import yaml
from prepare_sagemaker import BASE_SHA, EVIDENCE, copy_file

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import pack, unpack


def notebook(bundle_sha: str, manifest_sha: str) -> dict:
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
        """# ApexPINN M2/F6/mid: Kaggle checkpointed segments

Prepared, NOT RUN. Private notebook and private input dataset. Select GPU T4 x2 and enable Internet.
Only one T4 is visible to this runner. Python 3.12 and float64 are required. No CPU fallback.
Scientific budget stays 20 tuning trials, five inner calibration fits and five final members total.
F1-F5 and frozen S1 records are not rerun. A small restart oracle is separate validation compute.

Kaggle's 12-hour GPU session cap differs from your weekly GPU allowance. Use an 8-hour cooperative
segment target, leaving setup/export headroom. This is not a hard timeout guarantee: an in-flight
compiled operation or diagnostic must return before the process can pause. Save outputs promptly.
""",
    )
    cell(
        "markdown",
        """## 1. Verify and restore the private input
Upload `apexpinn_m2_f6_kaggle_v1.zip` as a private Dataset and attach it with Add Input.
Both an intact ZIP and Kaggle's expanded Dataset layout are supported. Verification precedes imports.
Dependencies and temporary inputs stay outside `/kaggle/working`, where only results are saved.
""",
    )
    cell(
        "code",
        f"""import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

assert sys.platform == 'linux' and os.environ.get('KAGGLE_KERNEL_RUN_TYPE'), 'Hosted Kaggle required'
os.environ.update(APEXPINN_REMOTE_RUNTIME='kaggle', CUDA_VISIBLE_DEVICES='0')
BUNDLE_SHA = '{bundle_sha}'
MANIFEST_SHA = '{manifest_sha}'
INPUT_ROOT = Path('/kaggle/input')
WORKSPACE = Path('/tmp/apexpinn_f6_' + MANIFEST_SHA[:16])
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

if not WORKSPACE.exists():
    archives = list(INPUT_ROOT.rglob('apexpinn_m2_f6_kaggle_v1.zip'))
    if len(archives) == 1:
        assert sha(archives[0]) == BUNDLE_SHA, 'Input ZIP hash mismatch'
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(WORKSPACE)  # Exact, locally verified ZIP bytes only.
    else:
        manifests = [p for p in INPUT_ROOT.rglob('bundle_manifest.json') if sha(p) == MANIFEST_SHA]
        assert len(manifests) == 1, 'Attach exactly one matching private input Dataset'
        source = manifests[0].parent
        WORKSPACE.mkdir()
        declared = json.loads(manifests[0].read_text())
        for name in ['bundle_manifest.json', *declared['files']]:
            target = WORKSPACE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
assert sha(WORKSPACE / 'bundle_manifest.json') == MANIFEST_SHA, 'Input manifest mismatch'
manifest = json.loads((WORKSPACE / 'bundle_manifest.json').read_text())
for name, item in manifest['files'].items():
    path = WORKSPACE / name
    assert path.is_file() and not path.is_symlink(), name
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
sys.path.insert(0, str(WORKSPACE / 'src'))
print('Verified input payloads:', len(manifest['files']))
""",
    )
    cell(
        "markdown",
        """## 2. Install the locked Python 3.12 environment and verify one T4
Setup does not fit a model. It downloads the frozen CUDA12 dependencies; Internet must be enabled.
Keep the same Kaggle environment image across segments. Do not upgrade source or dependencies mid-run.
""",
    )
    cell(
        "code",
        """import subprocess
from apexpinn.cloud.runtime import require_kaggle_environment

require_kaggle_environment()
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv==0.12.1'], check=True)
UV = shutil.which('uv')
assert UV, 'uv installation did not expose its executable'
subprocess.run([UV, 'sync', '--frozen', '--extra', 'cuda', '--no-dev', '--python', '3.12'],
               cwd=WORKSPACE, check=True)
PYTHON = WORKSPACE / '.venv/bin/python'
subprocess.run([str(PYTHON), '-c',
    "import jax; jax.config.update('jax_enable_x64', True); "
    "from apexpinn.train.cloud_checkpoint import require_remote_gpu; "
    "require_remote_gpu(); print(jax.devices(), 'float64:', jax.config.x64_enabled)"],
    cwd=WORKSPACE, check=True)
""",
    )
    cell(
        "markdown",
        """## 3. Select the segment and its previous checkpoint
First segment: set START_NEW_RUN=True and leave RESUME_ARCHIVE/RESUME_SHA256 empty.
Later segments: attach the previous successful notebook's outputs with Add Input, set START_NEW_RUN=False,
and supply its `f6_segment_....zip` path and printed SHA-256. Use only one execution per run identity.
An intact ZIP and its full-state checkpoint are required; parameters alone are insufficient.
Do not launch the same notebook interactively and in the background at the same time.
""",
    )
    cell(
        "code",
        """RUN_TRAINING = False  # Change only when you choose to start this remote segment.
START_NEW_RUN = False
SEGMENT_HOURS = 8
RESUME_ARCHIVE = ''   # e.g. /kaggle/input/.../f6_segment_0123456789ab.zip
RESUME_SHA256 = ''    # Copy the previous segment's printed archive SHA-256.
assert bool(RESUME_ARCHIVE) == bool(RESUME_SHA256), 'Provide both resume values'
if RUN_TRAINING:
    assert bool(RESUME_ARCHIVE) != START_NEW_RUN, 'Choose a new run OR an explicit resume'
""",
    )
    cell(
        "markdown",
        """## 4. Run one segment and export its saved state
For a browser-independent run, choose Save Version > Save & Run All after completing the parameters.
It starts a fresh session and executes this notebook top to bottom, not the current interactive process.
Training stdout/stderr is streamed and saved as text `.log` files alongside timestamped JSON records.
PAUSED means continue this same run next session. JOB_COMPLETE means F6 produced job.json and its report;
It does not establish the full T13/G2 gate. Optimizer/model/RNG/step/normalizers,
ledgers and logs are exported.
""",
    )
    cell(
        "code",
        """if RUN_TRAINING:
    from apexpinn.cloud.kaggle_runner import run_segment
    result = run_segment(PYTHON, WORKSPACE, Path('/kaggle/working'),
                         bundle_manifest_sha256=MANIFEST_SHA, hours=SEGMENT_HOURS,
                         resume=Path(RESUME_ARCHIVE) if RESUME_ARCHIVE else None,
                         resume_sha256=RESUME_SHA256 or None)
    print(json.dumps(result, indent=2))
else:
    print('NOT RUN. Setup only; set RUN_TRAINING=True when ready.')
""",
    )
    cell(
        "markdown",
        """## 5. Preserve outputs before the next session
Wait for the background version to finish successfully and verify its Output tab contains the
`exports/f6_segment_....zip`, receipt and summary. Download these, or attach that exact version's outputs
to the next notebook. Keep them private. Merely writing checkpoints on ephemeral working disk is not
off-session persistence. Do not use Quick Save as proof that the background job completed.

The segment target is cooperative, not an infrastructure time guarantee. Abrupt termination may lose
unpublished outputs and uncheckpointed time is Unknown. A failed notebook is not a completed F6 job;
preserve its logs and inspect it before another attempt. Do not reset tuning slots after a failure.
Completed outer forecast origins are cached. An interrupted inner evaluation/diagnostic may replay,
and recompilation occurs in a new session. Environment changes must fail closed rather than silently refit.
""",
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def prepare(base: Path, output: Path, *, repo: Path = Path(".")) -> dict:
    if output.exists():
        raise FileExistsError("choose a new versioned handoff directory")
    original = yaml.safe_load((repo / "configs/t13_cloud_v1.yaml").read_text())
    method = yaml.safe_load((repo / "configs/t13_kaggle_v1.yaml").read_text())
    changed = {key for key in original.keys() | method.keys() if original.get(key) != method.get(key)}
    if changed != {"method_id", "status", "training_location"}:
        raise ValueError("Kaggle declaration must preserve all scientific and numerical values")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        tmp = Path(temporary)
        restored, payload, stage = tmp / "base", tmp / "payload", tmp / "handoff"
        original_manifest = unpack(base, restored, expected_sha256=BASE_SHA, kind="input")
        stage.mkdir()
        names = [
            name
            for name in original_manifest["files"]
            if name.startswith(("data/gold_development_frozen_v1/", "data/cloud_baselines/"))
            and Path(name).suffix in {".parquet", ".json"}
        ]
        for name in [*names, *EVIDENCE]:
            copy_file(restored / name, payload / name)
        for name in ("pyproject.toml", "uv.lock", "README.md"):
            copy_file(repo / name, payload / name)
        for folder, pattern in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml")):
            for source in sorted((repo / folder).rglob(pattern)):
                copy_file(source, payload / source.relative_to(repo))
        if (
            digest(payload / "configs/decisions_v1.yaml")
            != original_manifest["files"]["configs/decisions_v1.yaml"]["sha256"]
        ):
            raise ValueError("frozen decisions changed")
        write_json(
            payload / "handoff.json",
            {
                "schema": "apexpinn.kaggle_input/v1",
                "base_sha256": BASE_SHA,
                "selection": "M2/F6/mid/main",
                "compute_status": "NOT RUN",
            },
        )
        archive = stage / "apexpinn_m2_f6_kaggle_v1.zip"
        manifest = pack(payload, [p for p in payload.rglob("*") if p.is_file()], archive, kind="input")
        spec = {
            "bundle_sha256": digest(archive),
            "manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
            ).hexdigest(),
            "status": "PREPARED_NOT_RUN",
            "selection": "M2/F6/mid/main",
        }
        write_json(stage / "train_m2_f6.ipynb", notebook(spec["bundle_sha256"], spec["manifest_sha256"]))
        copy_file(repo / "notebooks/kaggle/README.md", stage / "README.md")
        write_json(stage / "handoff.json", spec)
        stage.rename(output)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/colab_exports/apexpinn_t13_input_v1.zip"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.out), indent=2))


if __name__ == "__main__":
    main()
