"""Author/validate the thin Colab notebook with nbformat; never executes a notebook cell."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nb


def main() -> None:
    cells = []

    def markdown(text):
        cells.append(nb.v4.new_markdown_cell(text))

    def code(text):
        ast.parse(text)
        cells.append(nb.v4.new_code_cell(text))

    markdown("""# ApexPINN T13: train on Google Colab

This notebook prepares a remote GPU runtime and calls the repository runner. It has **not been executed**.
Local model fitting is prohibited. Use **Runtime > Change runtime type > GPU**, with a Google-hosted runtime.

G0/G1 and frozen S1 results remain unchanged. The default selection below is one complete M2/F1/mid job,
with its full 20-trial budget and five final members. It is a shard, not a G2 result. Expand the selection
for the full M1/M2, ablation and phi study. See `notebooks/colab/README.md` in the handoff for the method.
H2's separate within-known-circuit validation experiment is not implemented. The inner held-out
circuit is not an in-distribution test. This runner does not declare G2 acceptance automatically.

## 1. Mount persistent storage
Run these cells in Colab. No local-runtime connection or SSH tunnel to the laptop is needed.""")
    code("""import os
import sys
from pathlib import Path

try:
    from google.colab import drive, files
except ImportError as exc:
    raise RuntimeError('Use a hosted Colab GPU. Local execution is prohibited.') from exc
if sys.platform != 'linux' or not Path('/content').is_dir():
    raise RuntimeError('A hosted Colab runtime is required.')
drive.mount('/content/drive')
os.environ['APEXPINN_REMOTE_RUNTIME'] = 'colab'
""")
    markdown("""## 2. Select the input bundle and persistent run

Locally, `uv run --no-sync python scripts/prepare_colab.py` creates a code/data ZIP and receipt without
training. Upload them to `MyDrive/apexpinn/inputs/`. Paste the ZIP SHA-256 from that receipt below.
The bundle includes uncommitted working-tree code, accepted Gold and compact G1 forecasts.
The four excluded planning documents, credentials and frozen S1 run directories are omitted.

Keep `RUN_ID` unchanged when resuming the same run. Changed source, data, versions or GPU kind requires
a new declared run. This notebook never changes the frozen decisions or tuning budget.""")
    code("""DRIVE_ROOT = Path('/content/drive/MyDrive/apexpinn')
INPUT_BUNDLE = DRIVE_ROOT / 'inputs/apexpinn_t13_input_v1.zip'
BUNDLE_SHA256 = 'PASTE_SHA256_FROM_INPUT_RECEIPT'
RUN_ID = 't13_cloud_v1_run01'
EXPORT_TAG = 'checkpoint_01'

MODELS = ['M2']
FOLDS = ['F1']
PHIS = ['mid']
VARIANTS = ['main']
# Full study, same method/budgets:
# MODELS = ['M1', 'M2']
# FOLDS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
# PHIS = ['low', 'mid', 'high']
# VARIANTS = ['main', 'no_thermal', 'no_wear', 'constant_q', 'no_window']
if len(BUNDLE_SHA256) != 64 or any(c not in '0123456789abcdef' for c in BUNDLE_SHA256):
    raise ValueError('Paste the archive SHA-256 from the local input receipt.')
if any(not value or any(c not in
       'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value)
       for value in (RUN_ID, EXPORT_TAG)):
    raise ValueError('Run ID and export tag must contain only letters, digits, underscore or hyphen.')
RUN_ROOT = DRIVE_ROOT / 'runs' / RUN_ID
""")
    markdown("""## 3. Verify and restore the input to the remote VM

Large input reads happen on the VM after one verified copy from Drive. Checkpoints and results are
written directly to the persistent run folder on Drive. The bootstrap imports only standard-library
transport code from the already hash-verified ZIP, before installing the training environment.""")
    code("""import hashlib
import json

def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

if file_sha(INPUT_BUNDLE) != BUNDLE_SHA256:
    raise ValueError('Input archive hash mismatch.')
sys.path.insert(0, str(INPUT_BUNDLE) + '/src')
from apexpinn.cloud.transfer import unpack
WORKSPACE = Path('/content') / ('apexpinn_' + BUNDLE_SHA256[:12])
if not WORKSPACE.exists():
    unpack(INPUT_BUNDLE, WORKSPACE, expected_sha256=BUNDLE_SHA256, kind='input')
else:
    manifest = json.loads((WORKSPACE / 'bundle_manifest.json').read_text())
    for name, item in manifest['files'].items():
        if file_sha(WORKSPACE / name) != item['sha256']:
            raise ValueError('Remote workspace was changed; restore to a new workspace.')
print('Verified workspace:', WORKSPACE)
""")
    markdown("""## 4. Install locked Python 3.12 and CUDA dependencies

The Colab notebook kernel may have another Python version. Training uses the isolated project Python
3.12 environment. `uv sync --frozen` uses the supplied lockfile; it does not upgrade the research setup.
If dependency/device setup fails, stop and retain the error. No CPU training fallback is provided.""")
    code("""import shutil
import subprocess

subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
UV = shutil.which('uv')
subprocess.run(
    [UV, 'sync', '--frozen', '--extra', 'cuda', '--no-dev', '--python', '3.12'],
    cwd=WORKSPACE, check=True)
PYTHON = WORKSPACE / '.venv/bin/python'
subprocess.run([str(PYTHON), '-c',
    'import jax; jax.config.update("jax_enable_x64", True); '
    'from apexpinn.train.cloud_checkpoint import require_colab_gpu; '
    'require_colab_gpu(); print(jax.devices(), jax.config.x64_enabled)'], cwd=WORKSPACE, check=True)
""")
    markdown("""## 5. Inspect the job plan

This cell does not train. The main models have 20 unique inner-validation trials per fold/phi.
Five fixed-configuration calibration fits estimate member scales on inner validation, followed by
five separate outer-training fits. Calibration fits are explicitly timed and are not hidden tuning trials.
M1 ablations reuse main hyperparameters without another search.
M2 remains the user-selected forecasting engine, with M1 as comparator.""")
    code("""command = [str(PYTHON), 'scripts/run_models.py', '--out', str(RUN_ROOT)]
for flag, values in [('--model', MODELS), ('--fold', FOLDS), ('--phi', PHIS), ('--variant', VARIANTS)]:
    for value in values:
        command += [flag, value]
subprocess.run(command + ['--plan'], cwd=WORKSPACE, check=True)
""")
    markdown("""## 6. Verify restart behavior on the remote GPU

This is an actual tiny **cloud-only** training check, not S1 or G2 evidence. It compares uninterrupted
and interrupted/restored M1/M2 updates, including optimizer, offsets and RNG. Run it before the first
large job. A saved successful receipt can be inspected in a later session without rerunning it.""")
    code("""SELFTEST_ROOT = DRIVE_ROOT / 'checks' / (RUN_ID + '_' + BUNDLE_SHA256[:12] + '_restart')
if (SELFTEST_ROOT / 'selftest.json').exists():
    print((SELFTEST_ROOT / 'selftest.json').read_text())
else:
    subprocess.run(
        [str(PYTHON), 'scripts/run_models.py', '--out', str(SELFTEST_ROOT), '--self-test'],
        cwd=WORKSPACE, check=True)
""")
    markdown("""## 7. Train the selected jobs on Colab

**Running this cell starts model training on the remote GPU.** Completed/failed trial slots are retained.
An interrupted running fit resumes its compatible full-state checkpoint under the same run ID. Time lost
between the last checkpoint and interruption is reported as unknown, rather than hidden in runtime totals.

Keep this notebook open according to Colab's usage terms. Session lifetime, GPU availability and memory
are not guaranteed. If interrupted, repeat setup and rerun with the same unchanged identity.""")
    code("""LOG_ROOT = DRIVE_ROOT / 'logs' / RUN_ID / 'training'
subprocess.run(
    [str(PYTHON), '-m', 'apexpinn.cloud.process', '--cwd', str(WORKSPACE),
     '--logs', str(LOG_ROOT), '--', *command], check=True)
""")
    markdown("""## 8. Read saved status and export results

The result folder contains declarations, source snapshots, trial ledgers, calibration/final fits,
checkpoints, forecasts, metrics, S3/S4 diagnostics and a report. A partial shard reports G2 NOT ESTABLISHED.
G2 acceptance and user-owned P2 require a separate review of the complete study.

You can rerun the export after an interrupted job because committed checkpoints already live on Drive.
Use a new export tag for each snapshot; existing ZIPs are never overwritten.""")
    code("""report = RUN_ROOT / 'eval_physics.md'
if report.exists():
    print(report.read_text()[:6000])
else:
    print('No completed report yet. Inspect jobs/*/progress.json in the persistent run folder.')
EXPORT = DRIVE_ROOT / 'exports' / (RUN_ID + '_' + EXPORT_TAG + '.zip')
EXPORT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [str(PYTHON), '-m', 'apexpinn.cloud.transfer', 'pack-results',
     '--root', str(RUN_ROOT), '--out', str(EXPORT)], cwd=WORKSPACE, check=True)
print('Result archive:', EXPORT)
print('Receipt:', str(EXPORT) + '.receipt.json')
print('SHA-256:', file_sha(EXPORT))
""")
    markdown("""## 9. Return the artifacts

Download the ZIP and its receipt from Drive, or let Drive for desktop sync them to your laptop.
Then import to a **new** local folder, without training:

```powershell
uv run --no-sync python -m apexpinn.cloud.transfer unpack PATH_TO_RESULT.zip `
  --sha256 SHA_FROM_RECEIPT --kind results `
  --dest data/colab_imports/t13_cloud_v1_run01_checkpoint_01
```

Give the imported folder path to the agent for result review. For supported files, a connected Google Drive
link is another readback route. A notebook link alone does not give access to its live runtime.

Direct remote execution is an optional separate integration: Google's official [Colab MCP server](https://developers.google.com/colab/api/reference/mcp)
has endpoint `https://colaboratory.googleapis.com/mcp`. Enable its API and authenticate a supported client.
This notebook does not connect MCP. Checkpoint/Drive persistence is still required.

Sources: [Colab file I/O](https://colab.research.google.com/notebooks/io.ipynb),
[runtime limits](https://research.google.com/colaboratory/faq.html),
[JAX installation](https://docs.jax.dev/en/latest/installation.html).
Validation status: notebook structure and Python syntax can be checked locally; remote setup, training,
GPU restart parity and Drive round trips remain NOT RUN until you execute them on Colab.""")
    notebook = nb.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": "ApexPINN T13 Colab training", "provenance": []},
            "accelerator": "GPU",
        },
    )
    nb.validate(notebook)
    target = Path("notebooks/colab/train_folds.ipynb")
    target.parent.mkdir(parents=True, exist_ok=True)
    nb.write(notebook, target)
    print(f"Wrote and schema-validated {len(cells)} unexecuted cells: {target}")


if __name__ == "__main__":
    main()
