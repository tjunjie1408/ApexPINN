# Project status

Snapshot: **2026-11-09**. This page summarizes available evidence; it is not a new acceptance gate.

## Established work

- The synthetic study is complete and frozen. The [S1 report](../reports/s1_identifiability.md)
  retains failed recovery and identifiability outcomes rather than changing thresholds after the fact.
- The accepted development data pipeline and real B0-B3 benchmark are documented in the
  [data report](../reports/data_report.md) and [baseline report](../reports/eval_baselines.md).
- Saved real-data M2/mid artifacts through F5 have been imported and reviewed locally, including
  the separate recovered F1 lineage. This is a selected part of T13, not its full matrix.
- The separate clean-calibration F3 pilot has a completed local artifact-verification record.
  Its [method declaration](calibration_clean_v1.md) remains the original pre-execution statement.
- Remote execution has full-state checkpoints, trial accounting, timestamped logs and verified ZIP
  transport. A failed process, a recovered evaluation and a completed job retain distinct provenance.

The M2 completion records are local artifacts, not public download links:
`data/colab_imports/t13_m2_f{2,3,4,5}_mid_run01_completion_check.json` and
`data/colab_imports/calib_clean_m2_f3_mid_v1_completion_check.json`.
Their eventual publication needs a separate artifact release with stable identifiers and hashes.

## Current blocker

M2/F6/mid/main is pending. The [Kaggle handoff](../notebooks/kaggle/README.md) is prepared,
but the user's account is rate-limited during phone verification and cannot enable Internet.
The observed PyPI DNS/install failure occurred during setup, before training in that cell.
The platform has not supplied a verified unlock time. No F6 completion is claimed.

Kaggle T4 installation, real memory use, runtime and actual cross-session continuation still need
remote validation. Prepared code and mocked no-fit tests do not establish those outcomes.

## Remaining work

1. Complete and verify the selected M2/F6 run without restarting completed F1-F5 studies.
2. Resolve the remaining T13 comparisons, ablations, sensitivity work, within-known-circuit
   validation and the user's P2 decision. One completed mid-phi M2 sweep would not close all of G2.
3. Implement and verify the frozen predictor and pit-strategy simulator/benchmarks.
4. Resolve the holdout protocol, freeze the selected model and then perform the one-time temporal
   holdout evaluation. Existing development results do not stand in for this test.
5. Release reproducible source, selected data and model artifacts with cards, hashes and a tested
   download/load path. The intended GitHub/Hugging Face releases are not completed by local cleanup.

## How to describe the work

Supported scope today: a reproducible forecasting research pipeline using JAX, circuit-held-out
evaluation, checkpoint continuation and artifact provenance.

Claims that remain unsupported include production deployment, completed pit-stop optimization,
full end-to-end strategy validation and a general accuracy improvement over the strongest baseline.
After publication, add the actual release links and verified release scope before extending these claims.
