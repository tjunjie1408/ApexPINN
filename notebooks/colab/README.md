# Colab training and result handoff

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
