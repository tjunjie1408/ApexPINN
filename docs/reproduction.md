# Reproduction and execution

Use Python 3.12 and the committed `uv.lock`. Training runs on a supported remote GPU and is launched
by the user. Do not use a local CPU, local GPU, WSL or a local Colab backend for fitting.

## Local preparation

```bash
uv sync --frozen
uv run --no-sync pytest -q -p no:cacheprovider tests/test_kaggle_handoff.py
```

The selected file uses mocked execution and saved/toy artifacts without optimizer updates.
Do not substitute an unrestricted `pytest` run; some historical tests fit models.

Preview the pending Kaggle selection without creating a study or fitting:

```bash
uv run --no-sync python scripts/run_models.py --method configs/t13_kaggle_v1.yaml --out data/f6_plan --model M2 --fold F6 --phi mid --variant main --plan
```

This declares the unchanged budget of 20 tuning trials, five calibration fits and five final members
for the selected job. It does not verify a GPU, install a cloud environment or run the model.

## Remote execution

Choose a notebook from the [notebook index](../notebooks/README.md). Keep its input bundle, method,
source, data, package versions and device policy unchanged across compatible continuation sessions.
Check Python 3.12, the declared GPU and float64 before fitting. Never silently fall back to CPU.

Checkpoints retain model/optimizer state, offsets, RNG, step and loss normalizers. A resumed fit
uses its own committed state and keeps its original budget slot. Completed fits are reused;
failed trials do not receive unbudgeted replacements. Uncheckpointed lost time remains Unknown.

Kaggle segments export their state before a planned pause; verify the saved notebook Output before
replacing the session. A working-directory checkpoint alone is not off-session storage.
Colab uses Drive for persistent run state and versioned exports.

## Import saved results

Download the original result ZIP and matching receipt. Import into a new local directory:

```bash
uv run --no-sync python -m apexpinn.cloud.transfer unpack RESULT.zip --kind results --sha256 VERIFIED_ARCHIVE_SHA256 --dest data/imports/NEW_VERSION
```

The importer checks archive identity, payload hashes, sizes and paths before publishing a destination.
Keep the originals. Inspect process records, fit/trial counts, checkpoint identity, forecasts and physics
diagnostics before calling a job complete. `job.json` is the completed-job record; a ZIP by itself can
represent a partial run. Completion of a shard does not establish G2 or the user's P2 decision.

## Frozen evidence

Existing S1 records and their criteria remain unchanged. Generated reports and figures are rebuilt
from their builders, not hand-edited. The active Kaggle bundle is already hash-pinned; documentation
cleanup does not change the code inside an uploaded bundle. Rebuild into a new version if its runtime
code or dependencies must change. Do not replace an active input ZIP under an existing identity.
