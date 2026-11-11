# M2/F6 SageMaker handoff

Prepared, NOT RUN. This is a managed training job controlled by a notebook, not the quota-blocked
classic Notebook Instance product. No Studio GPU or domain is needed. A control kernel may run
locally because it only calls AWS APIs; all model fitting and the restart oracle run remotely.
Do not execute Docker builds, optimization or the training entry point on the workstation.

## Contents and scientific scope

- `train_m2_f6.ipynb`: separate setup, input upload, request preview, explicit launch and monitoring.
- `apexpinn_m2_f6_sagemaker_v1.zip` and receipt: self-contained code, frozen Gold, saved baseline
  predictions, G0/G1 and S1/P1 prerequisite evidence. No Drive mount is needed at runtime.
- `launch_spec.json`: exact ZIP SHA-256, account/region, M2/F6/mid/main and maximum runtime.
- `control/`: API helpers with no JAX dependency; `image/`: allowlisted scientific Docker context.
- `image-context.zip`: small code-only Docker build context for remote CodeBuild.
- `terraform/`: S3, ECR, log group, execution role and optional remote image builder definitions.
- `handoff_manifest.json`: hashes/sizes for every prepared handoff file except the manifest itself.

This new execution identity preserves all numerical settings from `t13_cloud_v1.yaml`. It uses
`t13_sagemaker_v1.yaml`; the original method/config/results are unchanged. The job performs a
small M1/M2 restart oracle, then20 tuning trials,5 calibration fits,5 final fits and the existing
forecast/S3-S4/report pipeline for M2/F6/mid/main. Here S3-S4 means scientific scenarios, distinct
from Amazon S3 storage. It does not rerun F1-F5 or complete the remaining T13 matrix/G2/P2.

## 1. Prepare infrastructure, then export its outputs

The agent has not applied Terraform or created IAM/S3 resources. User-run commands from the
handoff directory:

```powershell
aws login --profile apexpinn
terraform -chdir=terraform init
terraform -chdir=terraform plan -var=enable_remote_builder=true -out=setup.tfplan
# Review before applying. This creates persistent resources and IAM access; starts no compute.
terraform -chdir=terraform apply setup.tfplan
python -c "import pathlib,subprocess; pathlib.Path('infra.json').write_bytes(subprocess.check_output(['terraform','-chdir=terraform','output','-json','handoff']))"
```

Use a Python3.12 control environment with `requirements-control.txt`. Configure the named profile
through browser login, not pasted access keys. Terraform does not manage inputs, checkpoints or
training jobs. Do not put runtime research artifacts in Terraform state.
If the workstation login expires during a long run, use `aws login --profile apexpinn` again
to monitor. The managed job uses its execution role and does not depend on the workstation token.

## 2. Build the container remotely

The image uses Python3.12, the committed uv.lock with CUDA extras, and libgomp for LightGBM
imports. The mutable base-image tag is resolved during build; launch pins the resulting ECR
image digest. The Docker build has NOT RUN, and SageMaker driver/CUDA integration remains a
remote validation requirement. No notebook cell installs scientific dependencies on the GPU VM.

After applying the optional CodeBuild setup, choose to upload the code-only context and start
the remote build. Replace BUCKET, PROJECT, BUILD_ZIP_SHA and IMAGE_TAG from infra/launch outputs.
The code-only context upload and build are separate from the111MB-scale scientific input ZIP.

```powershell
aws s3 cp image-context.zip s3://BUCKET/build/BUILD_ZIP_SHA/image-context.zip --profile apexpinn --region ap-southeast-2
aws codebuild start-build --project-name PROJECT --source-location-override BUCKET/build/BUILD_ZIP_SHA/image-context.zip --environment-variables-override name=IMAGE_TAG,value=IMAGE_TAG,type=PLAINTEXT --profile apexpinn --region ap-southeast-2
aws codebuild batch-get-builds --ids BUILD_ID --profile apexpinn --region ap-southeast-2
aws ecr describe-images --repository-name apexpinn-m2-f6 --image-ids imageTag=IMAGE_TAG --query 'imageDetails[0].imageDigest' --profile apexpinn --region ap-southeast-2
```

Use a fresh immutable image tag tied to the prepared context SHA, and copy the resulting
`repository@sha256:DIGEST` into the notebook. CodeBuild compute costs extra and stops at the
30-minute build timeout; it is not a model fit. Do not start the build until you choose to use
AWS resources. No build has been started by this handoff. The operator must verify the uploaded
context checksum and review its source hash against `launch_spec.json` before starting CodeBuild.

## 3. Upload the scientific input and run the notebook

Keep the notebook, control folder, ZIP, launch_spec.json and infra.json together. Open
`train_m2_f6.ipynb` in a Python3.12 notebook environment. Preview the full request before setting
`UPLOAD_INPUT=True`. This uploads the exact ZIP using its SHA-256 S3 key and verifies the native
S3 checksum. Set `SUBMIT_TRAINING=True` only on your execution order. The defaults perform no
upload or GPU launch. One On-Demand `ml.g6.2xlarge` L4 is selected; no Spot/warm pool/endpoint.

The24-hour limit would be38.136USD compute-only at the observed1.589USD/hour Sydney Training
rate, excluding S3/ECR/logging/builds/storage/transfer/taxes. This is a price scenario, not a
runtime prediction or an account spend limit. Credit coverage of SageMaker must be checked on
your actual credits before spending. Available quota does not guarantee physical capacity.

## 4. Monitoring, durable results and recovery

CloudWatch `/aws/sagemaker/TrainingJobs` captures stdout/stderr. Immutable attempt `.log` files
and process JSON records are also under the S3 checkpoint prefix. Worker heartbeat JSON reports
actual elapsed wall time and UTC timestamps, independent of notebook timers. A heartbeat alone
does not establish optimizer progress; compare per-fit `progress.json` step counters.

The checkpoint path `/opt/ml/checkpoints` is explicitly synchronized with S3 by SageMaker.
No arbitrary notebook directory or Google Drive path is assumed persistent. Raw state includes
models, optimizer moments, offsets, RNG, normalizers, source/data/config identity and step.
SageMaker synchronization is asynchronous; abrupt loss can leave an incomplete latest generation.
Existing integrity checks refuse corrupt/missing committed state. Cross-job restoration and
signal/timeout handling remain NOT VERIFIED until remotely exercised.

On success, SageMaker uploads model.tar.gz containing the versioned result ZIP/receipt. Raw run
records and logs also remain under `runs/RUN_ID/checkpoints`. On failure, use CloudWatch's original
traceback and the raw checkpoint prefix. Final ZIP export is best effort under forced termination.
Never mark success solely from an uploaded ZIP or heartbeat: require exit0, job.json, report and
artifact hash verification. Downloading/validating the run is separate from scientific acceptance.

Repeated submission of the same job/request reuses its AWS record. The S3 run lock prevents
concurrent use of a checkpoint prefix. For a Failed/Stopped attempt only, explicitly set
RESUME=True and a new JOB_NAME while preserving RUN_ID, bundle and image digest. The worker
refuses mismatched state; no Colab-to-SageMaker checkpoint resume is claimed. Full numerical
restart equivalence is checked by the remote oracle before fitting. Lost uncheckpointed overhead
is Unknown. Inner/outer forecast evaluation can replay because it has no per-origin checkpoint.

After completion, training compute is released automatically. Stop any separately created Studio
apps yourself. S3/ECR/CloudWatch storage continues billing until explicitly cleaned up; Terraform
protects scientific storage against accidental destruction.

## Local preparation and validation

From the repository, prepare a NEW handoff folder with:

```powershell
uv run --no-sync python scripts/prepare_sagemaker.py --out data/sagemaker_exports/m2_f6_v1
uv run --no-sync pytest -q tests/test_sagemaker_handoff.py tests/test_execution_handoff.py
uv run --no-sync ruff check src/apexpinn/cloud/sagemaker_control.py src/apexpinn/cloud/sagemaker_worker.py scripts/prepare_sagemaker.py tests/test_sagemaker_handoff.py
```

These selected tests are explicitly no-fit. Do not execute the notebook, restart oracle, training
entry point or full suite locally. Keep private planning documents, credentials, Terraform state
and workstation MCP receipts out of all handoff archives. Infrastructure apply, CodeBuild, image
push, upload, GPU checks and training are NOT RUN by local validation.
