# M2/F6 in Studio JupyterLab

The Sydney deployment was retired at the user's request on 2026-11-09. Its existing infra.json
and setup ZIP identify a deleted domain/profile and are historical, not launch-ready. Inputs
remain verified in S3. No US replacement is deployed because organization policy denies both
checked US regions. A future approved destination needs a new deployment and handoff version.

Prepared, NOT RUN. Current route: Studio JupyterLab + uv + S3. The managed-job handoff is an
unused alternative documented in README_managed_jobs.md. Do not use its image/job launch cells
for this Studio run. The new input ZIP and notebook declare `t13_studio_v1.yaml` explicitly.

The numerical protocol is unchanged: M2/F6/mid/main,20 tuning trials,5 calibration fits,5 final
fits, shared forecast evaluation and physics diagnostics. A separate small M1/M2 checkpoint
restart oracle runs first. Completed F1-F5 results are preserved and not replayed.

## Prepare infrastructure and input

From the generated Studio handoff directory:

```powershell
aws login --profile apexpinn
terraform -chdir=terraform init
terraform -chdir=terraform plan -out=studio.tfplan
# Apply only after reviewing the plan and choosing to create resources.
terraform -chdir=terraform apply studio.tfplan
python -c "import pathlib,subprocess; pathlib.Path('infra.json').write_bytes(subprocess.check_output(['terraform','-chdir=terraform','output','-json','handoff']))"
```

No GPU app is started by default. These commands are user-run instructions, not executed actions.
Upload the prepared ZIP to `s3://BUCKET/inputs/BUNDLE_SHA256/BUNDLE_NAME`, using launch_spec.json
for exact names/hash and infra.json for the bucket. Example user-run upload:

```powershell
aws s3 cp BUNDLE_NAME s3://BUCKET/inputs/BUNDLE_SHA256/BUNDLE_NAME --checksum-algorithm SHA256 --profile apexpinn --region ap-southeast-2
```

The Studio setup verifies the complete downloaded ZIP SHA-256 before extracting code/data.
The input ZIP includes frozen Gold and saved baselines; it needs no Drive connection, custom
Docker image, ECR or CodeBuild. Private planning files, credentials and workstation state are
excluded. The separate `control/` package contains only setup, logging, storage and runtime helpers.

## Start Studio when ready

Review/apply a new plan with `-var=start_compute=true` only when you want paid compute to start,
or use the Studio Console to start the prepared private space. Selected default is one L4 on
**ml.g6.2xlarge**,8vCPUs/32GiB. Official observed Sydney Studio rate is1.5889USD/hour; a hypothetical
16-hour run is Inferred25.4224USD compute-only. Runtime is unmeasured. EBS/EFS/S3 and other costs
are extra; credits are not a spending cap. SageMaker credit eligibility still needs checking.

Upload train_m2_f6.ipynb, launch_spec.json, infra.json and the whole control/ folder together.
Use a fresh Studio notebook kernel for a new handoff version. It may have a different Python
version; uv creates the scientific Python3.12 CUDA environment on the persistent space volume.
No fitting occurs during setup/GPU checks. The scientific CLI refuses local/CPU/T4/float32 use.

Keep RUN_TRAINING=False until your execution order. The training cell stays in the foreground,
keeping the kernel busy while the child process runs. Browser disconnection does not itself stop
the remote app, but idle policy/app maintenance can interrupt it. Actual behavior is NOT VERIFIED
until remote execution. The24-hour process timeout does not shut down the Studio GPU app.

## State, logs and S3 backup

Full model/optimizer/offset/RNG/normalizer state and step counters are committed on the persistent
space volume using the existing checkpoint implementation. Attempt stdout/stderr and JSON
outcomes are saved there too. Inspect logs and progress.json step/timestamp movement rather than
the notebook timer. Do not interpret a running app as proof of optimizer progress.

Every5 minutes and at exit, the runner copies run state, validates checkpoint/model publication,
packs a hash-manifest ZIP and uploads an immutable S3 snapshot. Only after the archive upload
succeeds is `studio-runs/RUN_ID/latest.json` updated. A failed/incomplete snapshot keeps the
previous S3 pointer. Restore verifies the ZIP hash, every payload and execution identity before
publishing a new local state directory. Existing local state is never overwritten or rolled back.

Resume keeps the same input/source/method and Studio context. A local lock prevents duplicate
execution on the same app. Failed fits keep their slots. Lost unsynced work/time is Unknown;
forecast evaluation can replay because it is not checkpointed per origin. Snapshot transport,
CUDA integration, idle behavior and app restart recovery remain NOT VERIFIED remotely. S3 backup
failures are printed; local EBS remains the primary current state until a snapshot succeeds.

No managed-job automatic checkpoint sync or automatic compute teardown is assumed in Studio.
No model training, inference, image build, infrastructure apply or upload is performed by local
tests. To regenerate locally, use a NEW output directory:

```powershell
uv run --no-sync python scripts/prepare_sagemaker.py --runtime studio --out data/sagemaker_exports/studio_m2_f6_v1
```

## Completion reminder: STOP/DELETE the Studio app

After exit0, job.json and the formal report exist, verify/download the final S3 archive and
receipt/pointer. Then **STOP/DELETE the Studio JupyterLab app and verify Deleted**. Closing its
tab or browser is insufficient. The notebook prints this reminder before and after training,
and has a final cleanup section. The agent will repeat it when reviewing the completed F6 run.

Delete the space only after backup if you want to remove EBS storage too. That deletion removes
its files. Domain EFS and S3 storage can remain separately billable; preserve research archives
and review storage cleanup explicitly. The Terraform space/S3 prevent_destroy guards protect
data. Nothing automatically deletes the app, space, checkpoints or archives.
