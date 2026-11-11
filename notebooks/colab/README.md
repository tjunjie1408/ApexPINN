# Colab training and result handoff

## Current selection

The pending selected job is M2/F6/mid/main. M2/mid artifacts through F5 are already imported and
reviewed locally. Use the [notebook index](../README.md) to choose the current route. Kaggle is the
latest prepared alternative; the Colab L4/Drive route remains available if explicitly selected.
Do not rerun completed folds or the clean-calibration pilot as routine notebook setup.

## Historical M1/F1 handoff and recovered M2/F1 result

The recorded M1/F1 handoff used `train_next_m1_v4.ipynb`, `apexpinn_t13_code_update_v4.zip` and its receipt in
Drive `apexpinn/inputs`. It reuses the original input ZIP parts. The notebook verifies both archives,
restores into a new workspace and selects M1/F1/mid/main with run ID `t13_m1_f1_mid_patch01`.
This describes the earlier handoff, not an instruction to launch another M1/F1 run.

The update contains the existing recovered M2/F1/mid evaluation and its original forecasts. The
collector accepts it through `--recovered-review` after checking method/config/Gold, numerical source,
fit budgets, diagnostics and forecast identity. Repeated job identities are rejected. The original
failed subprocess and the recovered evaluation keep their separate provenance.

Training and restart commands use `apexpinn.cloud.process` to preserve stdout/stderr and per-attempt
exit records in `MyDrive/apexpinn/logs/RUN_ID/`. Checkpoint steps also appear in this log. A failed
command prints its process-record path. Each invocation receives new log files.

Prepare a separately versioned update from saved artifacts with a small file-packaging command:

```powershell
uv --no-cache run --no-sync python scripts/prepare_colab_patch.py `
  --recovery data/colab_imports/t13_run01_checkpoint_01_report_with_s4/review.json `
  --original-run data/colab_imports/t13_cloud_v1_run01_checkpoint_01 `
  --out data/colab_exports/NEW_HANDOFF --tag NEW_VERSION
```

Use lowercase letters, digits, underscore or hyphen for the version. Existing handoff directories
are refused. Upload the generated notebook, update ZIP and receipt to Drive inputs. This command
copies source and saved forecasts. Model execution and long compute tasks require Colab and a
separate user order. H2 and the remaining scientific T13 work follow the recorded project plan.

Use [train_folds.ipynb](train_folds.ipynb) on a Google-hosted GPU. The agent prepares code and reviews
returned artifacts locally. The user launches every training cell. No local CPU/GPU/WSL fitting,
fitting tests, local-runtime connection or notebook execution is authorized.

## Prepare once on the laptop, without training

```powershell
uv --no-cache run --no-sync python scripts/prepare_colab.py
```

This reads the completed G1 studies, exports their forecast records losslessly to compact Parquet,
and writes `data/colab_exports/apexpinn_t13_input_v1.zip` plus its `.receipt.json`. It packages the
current working-tree Python code, locked environment, method/config, verified development Gold and
small prerequisite receipts. It does not rely on a GitHub push, copy the large original G1 JSON,
or include `GLOSSARY.md`, `PLAN.md`, `PRD.md`, `PROJECT.md`, credentials or frozen S1 run directories.
The original files remain intact. The first export reads about one GB of existing JSON per phi;
this is file processing, not fitting. Existing compatible exports are verified and reused.

Upload the ZIP and receipt to `MyDrive/apexpinn/inputs/`. Open the notebook with Colab's Upload notebook
action, select a hosted GPU, and paste the receipt's archive SHA-256 into the settings cell.
The notebook installs a locked Python 3.12 virtual environment even if its host kernel is newer.
It refuses a CPU fallback and keeps data reads on the remote VM after one verified transfer.

## Declared T13 method

`configs/t13_cloud_v1.yaml` is an implementation declaration before cloud results, not an optimized
method or a revision to frozen S1/G0/G1. Approved O1-O8 values remain unchanged.

- Main M1/M2: 20 unique trials per model/fold/phi; failures retain budget slots. M1 uses five learning
  rates crossed with two residual weights and two collocation densities. M2 uses 20 learning rates.
- The next circuit group is inner validation. Normalization is refitted using inner training circuits
  only. Final training uses outer training circuits; external test circuits never enter fitting/tuning.
- Each selected configuration has five separate inner calibration fits, used only to estimate each
  final member's predictive scale from inner primary-horizon residual RMSE. These are **additional,
  explicitly timed calibration fits**, not hyperparameter trials. Five outer refits are timed separately.
- M1 ablations remove thermal residual, remove wear/monotonicity residuals, replace q and its context
  by a training-only constant, or set gamma to exactly zero with finite raw parameters. They reuse main
  hyperparameters without new searches. M2 remains the structural comparator.
- Exact mixture variance is `mean(component_sigma**2) + variance(component_means)`. It need not exceed
  the largest component variance. The user confirmed this correction to the older PLAN inequality.
  No interval is inflated to meet that inequality.
- Physics predictive scales use inner-validation residuals; historical G1 baseline scales use training
  residuals. Probability comparisons have this explicit condition difference. Neither is a measured
  tire-noise parameter or proof of calibration.
- Parameter priors and physical equations are reused without editing their frozen source. M1 remains
  the original linear latent formulation; M2 is the user-selected P1 engine. No new P2 choice is made.

The default notebook selection is M2/F1/mid/main, with the full budget. It is a shard, not G2 acceptance.
The full forecast/ablation selection is 108 jobs, 720 tuning trials, 540 calibration fits and 540 final
fits (Inferred directly from the declared Cartesian products). Runtime, memory, storage and GPU access
are not guaranteed. Inspect the first shard before expanding the selection. Do not reduce budgets or
change thresholds based on observed failures; declare a new method/version/output for a revision.

The runner writes exact-horizon forecasts, exact-mixture scores, rejection audits, paired CIs against
G1, S3 strata, representative continuous-grid residual/S4 checks and physical parameters. H2's separate
known-circuit validation-stint experiment is **NOT IMPLEMENTED**; a held-out circuit is not ID validation.
Neither a successful shard nor the full forecast sweep alone establishes all project hypotheses/G2.
P2 and scientific gate acceptance remain review steps.

## Resume and persistence

Use the same run ID and source/input/environment identity to resume. Checkpoints on Drive include
model parameters, training offsets, optimizer moments/count, RNG, next step and loss normalizers.
Generations are immutable and checksum-verified; a pointer is published only after the generation.
Completed/failed fit and trial slots are retained. Corrupt/incompatible checkpoints stop execution;
they must not be silently restarted. Uncheckpointed interruption overhead is unknown and explicitly
excluded from recorded elapsed compute totals. Different GPU kinds or package versions need separate
outputs; cross-device bitwise continuation is not claimed.

Run the notebook's small remote restart oracle before a large job. Its model/offset/optimizer/RNG
parity is **NOT RUN** until the user executes it on Colab. It is a crafted check, not a frozen S1 rerun.

## Return results without exposing a runtime

The notebook exports a versioned ZIP and receipt under `MyDrive/apexpinn/exports/`. It can export a
partially completed run because committed checkpoints already live on Drive. Use a new export tag
for each snapshot. Download both files or let Drive for desktop sync them, then import locally:

```powershell
uv --no-cache run --no-sync python -m apexpinn.cloud.transfer unpack PATH_TO_RESULT.zip `
  --sha256 SHA_FROM_RESULT_RECEIPT --kind results `
  --dest data/colab_imports/t13_cloud_v1_run01_checkpoint_01
```

The importer validates paths, collisions, schema, sizes and all hashes before publishing a new
destination. It refuses existing destinations. Give the agent that folder path for review. A connected
Google Drive file link can also support artifact retrieval; a notebook URL alone is not a live-runtime
connection. Large binaries are better transferred as the ZIP, rather than pasted into chat.

## Optional direct Colab connection

Google documents a remote [Colab MCP service](https://developers.google.com/colab/api/reference/mcp)
at `https://colaboratory.googleapis.com/mcp`, with runtime/session discovery and code execution.
It requires [API enablement](https://docs.cloud.google.com/mcp/enable-disable-mcp-servers),
[authentication](https://docs.cloud.google.com/mcp/authenticate-mcp) and a compatible client.
MCP setup is separate from this repository. Creating the notebook does not configure that connection.
Keep account credentials and client configuration outside the repository and handoff archives.

For a long in-flight MCP execution, Google's API documents reattaching with the same execution ID.
Do not reissue a completed training call with that ID: the API may start a new execution. Use the
runner's immutable slots and persistent checkpoints either way. Drive persistence remains necessary
when a Colab runtime ends. Use account login, not credentials pasted into chat.

The current workflow is user-run notebooks plus Drive artifacts. The agent uploads code/data and
downloads/imports results when requested. This needs no agent runtime connection; it does not authorize
agent-managed training or runtime allocation.

Official references: [Colab file I/O](https://colab.research.google.com/notebooks/io.ipynb),
[runtime limits](https://research.google.com/colaboratory/faq.html),
[local-runtime semantics](https://research.google.com/colaboratory/local-runtimes.html),
[JAX GPU installation](https://docs.jax.dev/en/latest/installation.html).

## Verification boundary

Notebook schema, cell syntax, transport, guard behavior, serialization and mocked orchestration may be
checked locally. No local fitting test is allowed. Remote installation/GPU behavior, real optimizer
continuation parity, actual T13 forecasts and Drive transfer remain NOT RUN until user execution.
Original research records and local planning documents must remain unchanged.

## Remaining M2/mid folds with a compute-unit budget

`train_m2_remaining_mid.ipynb` defaults to F4 and launches exactly one original T13 v1 job.
After exporting/reviewing that fold, select F5 and then F6 in separate executions. Each has its own
run ID and 20 tuning trials, five calibration fits and five final members. It reuses the existing
v4 input overlay and source-matched restart receipt; no new restart fits or training-source changes.

Refresh AVAILABLE_UNITS from your account before each new fold. The user-reported 100 units at
1.94 units/hour is 51.55 L4 hours. A 12-hour-per-fold planning assumption gives 69.84 units for
three folds, leaving 30.16; 8/16-hour scenarios give 46.56/93.12 units. These are assumptions,
not runtime bounds. The editable 20-unit reserve is an operating preference, not a billing cap.
Setup, compilation, interruptions and idle allocated time can add cost. The full 104-job remainder
is not scheduled by this notebook. See `reports/figures/review_20261107_v1/budget.json`.

The launch cell skips an existing completed job and uses a same-VM Linux `flock` to reject a
concurrent launch of that run ID. It cannot coordinate separate VMs. Use one runtime per run ID.
Training, checkpointing and logging remain on Colab/Drive. Export validation checks the selected
job before packing. Release the runtime manually after the export is safe; no automatic next job
or runtime action is performed. Local checks are structural/AST and mocked no-fit execution only.

## Separate clean M2 calibration pilot

`calibration_clean_v1.ipynb` prepares a post-hoc, separately versioned experiment declared in
`configs/calibration_clean_v1.yaml` and explained in `docs/calibration_clean_v1.md`. Leave the current
M2/F3/mid T13 v1 run unchanged. The new notebook refuses to launch training until that run's completed
job record exists. Its code ZIP must be present in `MyDrive/apexpinn/inputs` before setup.

The default pilot reserves F3 for testing, F4 for tuning validation and F5 for independent calibration.
It requires 20 new tuning trials and five final fits, each with 1000 updates, on user-launched Colab
L4/float64/Python 3.12. The same five final models predict calibration and test data; no refit follows
calibration. Four horizon-specific factors transform the entire mixture around its unchanged mean.
Compare raw/calibrated coverage, width and CRPS by circuit and horizon. No coverage guarantee or
T13 v1/G2 completion is implied. Real execution and empirical effectiveness remain **NOT RUN**.

The runner checkpoints fits and reuses hash-verified forecast bundles. Numerical evaluation is saved
before plotting; `scripts/run_calibration.py --out RUN_DIRECTORY --report-only` rebuilds the saved-score
report without loading models. Restore any missing or corrupted source artifact before recovery.
Use a new output directory when method, source, data or environment identity changes.

## Saved M1 offset and shape diagnostic

`diagnose_m1_offsets.ipynb` is a prepared, unexecuted notebook for the completed
`t13_m1_f1_mid_patch01` run. Open it on a hosted L4 GPU. It reuses the approved input parts and
the separate `apexpinn_m1_diagnostic_v1.zip` code update in `MyDrive/apexpinn/inputs`.
The user launches it; the agent does not allocate a runtime or execute models.

The notebook calls `scripts/diagnose_m1.py`. It loads the five calibration and five final model
checkpoints and their training offsets, verifies source/package/data/state identities, and compares
the existing training residuals with residuals after closed-form offset profiling. Optimizer state
is deserialized as data; no gradients or optimizer updates are executed. Plots show the offset-error
decomposition and a fixed training-curve example. Profiled training errors use all training labels
and must not replace held-out forecasts or be reported as new fitted models.

Reports, per-member JSON, plots and the completion receipt go to a new folder under
`MyDrive/apexpinn/diagnostics/m1_offsets_f1_mid_v1`. Original checkpoints and scores remain intact.
The final cell exports only this diagnostic and its process logs. Source or package mismatches stop
the command; do not respond by rerunning training. If all ten per-member JSON files exist and only
plotting failed, recover the report from those files instead of repeating model inference.

Rebuild a handoff locally without model execution:

```powershell
uv run --no-sync python scripts/prepare_m1_diagnostic.py --out data/colab_exports/NEW_DIAGNOSTIC_HANDOFF
```

Local verification covers pure offset arithmetic, guard ordering, notebook syntax and payload hashes.
Saved-model restoration, GPU inference and chart rendering remain **NOT RUN** until user execution.

The original v1 handoff completed its ten numerical checks on 2026-11-05, then failed while importing
Matplotlib because Colab exported an inline backend into the isolated environment. That historical
handoff is preserved. The current renderer sets `MPLBACKEND=Agg` before importing Matplotlib and
writes numerical evidence separately from plot completion. Regenerate a new handoff before future
model diagnostics; the already-uploaded v1 ZIP retains its original bytes.

To recover a plotting-only failure, download all ten member JSON files and the original process exit
receipt. From the current checkout, run this **artifact-only** command, which imports no JAX and
does not load models or checkpoints:

```powershell
uv run --no-sync python scripts/diagnose_m1.py --render-only `
  --saved PATH_TO_SAVED_MEMBER_JSONS --run PATH_TO_IMPORTED_COMPLETED_M1_RUN `
  --out NEW_REPORT_DIRECTORY --process-status PATH_TO_ORIGINAL_EXIT_RECEIPT
```

Recovery verifies each fit/batch identity and the offset/shape arithmetic before rendering. Its
`RECOVERED_DIAGNOSTIC_COMPLETE` receipt retains the original failed exit. It does not convert that
process into a successful execution or replace any forecast score.
