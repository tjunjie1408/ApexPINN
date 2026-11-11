# Notebook index

Notebooks are thin execution wrappers. The user launches remote fitting; the package and CLI own
the model implementation. Local checks must not execute fitting cells.

| Route | Entry point | Current purpose |
| :--- | :--- | :--- |
| Kaggle | [F6 notebook](kaggle/train_m2_f6.ipynb), [instructions](kaggle/README.md) | Pending M2/F6/mid, one T4, Python 3.12 and float64; Internet verification blocked at the recorded checkpoint |
| Colab | [Remaining mid-fold notebook](colab/train_m2_remaining_mid.ipynb), [instructions](colab/README.md) | Existing L4/Drive workflow; select one fold explicitly and preserve its run identity |
| Calibration | [Clean F3 pilot](colab/calibration_clean_v1.ipynb) | Separate declared pilot; completed local evidence exists, so do not rerun it as routine setup |
| Diagnostics | [M1 offset diagnosis](colab/diagnose_m1_offsets.ipynb) | Saved-model investigation, separate from a new scientific tuning budget |
| Optional AWS references | [Infrastructure index](../infra/README.md) | Prepared alternatives; AWS is excluded from the current F6 route |

The fixed F2/F3 notebooks and original `train_folds*.ipynb` files are retained for their recorded
handoff/recovery lineage. They are not additional pending runs or permission to repeat completed folds.
Remote runs consume quotas even when the browser timer is stale. Inspect timestamped process logs
and fit progress, rather than allocating another session to refresh a display.

The separate GitHub/Hugging Face publication workflow is described in [publication](../docs/publication.md).
