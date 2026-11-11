# M2/F6 on EC2, version 1

Current deployment checkpoint, 2026-11-09: code/bundles and five IAM/security resources are ready,
but EC2 returned InsufficientInstanceCapacity for g6.2xlarge in Sydney2a,2b and2c. All retry
processes were stopped. No instance/root volume is allocated; setup, GPU checks and training
are NOT RUN. Inputs/control ZIPs are verified in S3. Local launch.tfvars.json now explicitly
sets launch_instance=false. A later user-ordered retry needs a fresh plan with it enabled;
do not apply an old failed saved plan. See data/aws_setup/ec2_deploy_20261109/ for raw evidence.

Prepared EC2 route for M2/F6/mid/main. Original Colab, synthetic and completed F1-F5 results remain
unchanged. The new method t13_ec2_v1.yaml changes execution metadata only:20 tuning trials,
5 calibration fits,5 final fits and the existing forecast/physics/report pipeline. A separate
small M1/M2 checkpoint restart oracle runs first, only when the training service is user-launched.

Observed Sydney Linux On-Demand g6.2xlarge compute price is1.27107USD/hour, source
data/aws_setup/ec2_sydney_price_20261109_raw.json. Inferred24h compute cost is30.50568USD =
24 * 1.27107, before EBS/network/public-IPv4 costs. This is a budget calculation, not a measured
runtime estimate. Capacity remains unverified until EC2 allocation succeeds.

## Setup and launch boundary

Prepare a NEW handoff folder with scripts/prepare_sagemaker.py --runtime ec2. The scientific ZIP
and small control ZIP have separate SHA256 manifests. Upload them under hash-addressed inputs
keys. Apply only a reviewed Terraform plan in infra/ec2; the retired Studio recipe is not used.
First boot does setup/GPU checks only, writes bootstrap logs to EBS/S3, then stops the host.
Verify setup.json or the final bootstrap log before ordering training. No notebook or Drive mount.

## User-launched training

Use the instance ID from the deployment receipt; placeholders below are NOT executed commands.
Start the same instance, wait for SSM to be online, then explicitly submit start_training.json.
The service is disabled on boot and never starts merely because the host is restarted.

```powershell
aws ec2 start-instances --instance-ids INSTANCE_ID --profile apexpinn --region ap-southeast-2
aws ssm describe-instance-information --filters Key=InstanceIds,Values=INSTANCE_ID --profile apexpinn --region ap-southeast-2
aws ssm send-command --instance-ids INSTANCE_ID --document-name AWS-RunShellScript --parameters file://start_training.json --profile apexpinn --region ap-southeast-2
```

The model environment uses Python3.12, JAX GPU and float64, with no local/CPU/T4 fallback. Full
checkpoints include optimizer, RNG, offsets, step and normalizers. Model fitting runs only on EC2.
Each command attempt has a persistent .log and JSON process outcome. Read them with SSM using
`journalctl -u apexpinn-train.service -n 100 --no-pager` or from /opt/apexpinn/state/m2-f6-mid-v1/logs.
S3 ec2-runs/m2-f6-mid-v1/status.json provides step/time monitoring without downloading all models.

## Backup and resume

EBS is primary current state. A verified S3 snapshot is attempted every300s and at exit. Latest
pointer is published only after immutable archive upload succeeds. Restore verifies every file,
source/input/method and host identity; it never overwrites existing local state. Resume by
starting the SAME instance/service, unchanged source/spec and run ID. Replacement-instance
recovery requires a separate reviewed handoff because the transport identity includes instance ID.
Failed/completed trials keep budget slots. Lost work since the last committed checkpoint or
successful S3 snapshot is Unknown. Forecast evaluation may replay; it is not saved per origin.

Do not treat GPU memory use, a running service or a status file as scientific completion. Require
exit0, job.json, complete formal report/diagnostics and local archive verification. G2/P2 remain
separate decisions. S3 transport and EC2/CUDA behavior need real remote verification.

## Finish and remove compute

The service stops the host after exit/failure; confirm EC2 actually reports stopped. Verify final
S3 results before terminating. Then disable termination protection deliberately, terminate the
instance and verify root EBS removal. EBS persists while stopped and remains billable. Closing a
browser/terminal does not stop compute. Preserve S3 scientific archives and original results.
