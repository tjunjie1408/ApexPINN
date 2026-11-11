# Kaggle M2/F6/mid handoff

Prepared, NOT RUN. User-launched remote training only. No local fitting and no AWS resources.
The user reports successful Colab Pro linking and a Kaggle bonus-hours confirmation. The account's
actual remaining weekly allowance has not been independently inspected. Do not promise a completion time.

## First segment

1. Create a **private Kaggle Dataset** from `apexpinn_m2_f6_kaggle_v1.zip` in the prepared handoff.
   Keep the uploaded dataset and all result notebooks private. Kaggle may expand the ZIP; either layout works.
2. Create a private notebook and import `train_m2_f6.ipynb`. Add that dataset as an Input.
3. Enable Internet and select GPU T4 x2. The notebook exposes only GPU 0 to JAX. Two 16 GB GPUs do not
   provide one 32 GB memory pool or automatically accelerate this single-device algorithm.
4. Run the input/environment setup cells. Require one NVIDIA T4, Python 3.12 and `float64: True`.
   This checks installation and device availability, not F6 memory fit or end-to-end speed.
5. Set `START_NEW_RUN=True`, `RUN_TRAINING=True`, `SEGMENT_HOURS=8`; leave resume fields empty.
   Stop any interactive training before submitting a background copy.
6. Use **Save Version > Save & Run All** for a new background session. Keep its image unchanged across
   segments. The first segment performs a small remote restart oracle before selecting only M2/F6/mid/main.
7. When the version finishes, verify the Output tab contains its checkpoint ZIP, receipt and summary.
   Download them or keep that exact output version attached to the next notebook.

## Resume

1. Attach the previous successful notebook's outputs using Add Input. If Kaggle expands its ZIP,
   download the ZIP and upload it as a file without extraction, or preserve the original ZIP in a private
   dataset. The runner deliberately requires the original archive and SHA-256 for result restoration.
2. Set `START_NEW_RUN=False`; set `RESUME_ARCHIVE` to the attached `f6_segment_....zip` and
   `RESUME_SHA256` to its printed hash. Keep code, frozen dependencies, method, run path and GPU policy unchanged.
3. Submit the next background version. Full optimizer/model/RNG/offset/step/normalizer state resumes.
   Completed fits and trials are reused. There are 20 total tuning slots, five calibration fits and five
   final members across the entire F6 run, not a new budget per segment.
4. A `PAUSED` segment is an operational checkpoint, not a finished scientific job. `JOB_COMPLETE` requires
   `run/jobs/M2_F6_mid_main/job.json`. Share the final ZIP/receipt for import and verification. F6 alone
   does not complete the full T13 matrix or establish G2/P2.

## Limits and monitoring

- Kaggle documents a 12-hour CPU/GPU session limit and up to 20 GB saved notebook output. The 8-hour
  target is a cooperative pause, not a hard timeout. An in-flight compiled operation/diagnostic can overrun.
  Installation/export also take time. A new runtime must compile again. Actual F6/T4 duration is unknown.
- Check fit `progress.json`, trial JSON and timestamped `.log`/JSON process records. GPU allocated memory
  and a frontend timer alone do not prove optimizer progress. Text logs, checkpoints and exported receipts
  serve different purposes; retain all of them.
- Completed outer forecast origins are persisted and reused. Interrupted inner evaluations or diagnostics
  may replay. Uncheckpointed interruption overhead remains Unknown. Never restart exhausted failed slots.
- `/tmp` contains the disposable workspace/environment; `/kaggle/working` holds raw state and result ZIPs.
  Notebook outputs are persistent only after the background version is successfully saved and verified.
  An abrupt platform termination can prevent final export. This route does not guarantee zero lost work.
- Linking Colab Pro grants quota eligibility; it does not mount Google Drive or automatically transfer
  checkpoints. No Drive credentials, private planning documents or workstation receipts enter this bundle.
- The CUDA12/JAX stack, T4 memory use, remote restart oracle, planned pause/resume and Kaggle output
  publication require remote validation. All remain NOT RUN until the user executes them.

Sources: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks),
[GPU allowance](https://www.kaggle.com/docs/efficient-gpu-usage),
[current hardware announcement](https://www.kaggle.com/discussions/product-announcements/735239),
[Colab Pro bonus](https://www.kaggle.com/blog/level-up-your-compute-more-gpu-hours-on-kaggle-wit).
