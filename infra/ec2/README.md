# EC2 L4 and retained S3 inputs

2026-11-09 checkpoint: IAM/profile/security setup exists, but all three Sydney g6.2xlarge
allocation attempts failed for capacity. No instance or project root EBS is allocated.
Retry processes are stopped and local launch vars are set to false. Do not treat installation,
hardware checks, automatic stop or fitting as remotely verified. Use a fresh reviewed plan
for any later user-ordered retry. The ready no-compute plan has no resource changes.

Sydney/account985882743884 only. New EC2 state is independent of the retired Studio state.
No S3 bucket is created or destroyed here. This recipe reads the retained inputs bucket and
limits writes to ec2-runs/m2-f6-mid-v1. No inbound SSH/Jupyter ports, keys or static credentials.
SSM uses AmazonSSMManagedInstanceCore plus the scoped S3 role policy.

launch_instance=false prepares IAM/security resources only. With launch_instance=true, one
On-Demand g6.2xlarge is allocated in the selected default subnet. Advertised AZ support and
quota8 are not physical capacity guarantees. Do not allocate multiple replacements at once.
The AMI is pinned to AWS's verified GPU-driver image ami-085f952c300f89090, owner898082745236.
Its root EBS is encrypted100GiB gp3. /opt/apexpinn must reside on that root volume, never instance
store. Stop/start keeps it; termination removes it. Termination protection is enabled by default.

Bootstrap downloads the exact hashed control ZIP, verifies every payload, installs the frozen
Python3.12 CUDA environment and checks one L4/float64. No oracle or fitting runs during setup.
Bootstrap then stops the instance, including on failure. Logs are on EBS and uploaded to S3.
apexpinn-train.service is installed but not enabled or started. After a separate user order,
start the host and service through SSM. Training is independent of a browser/terminal session.

The training service has a24h runtime limit and stops the host on exit/failure. A26h boot timer
also stops an idle host. These are safeguards, not a guaranteed spending cap; AWS state must
still be monitored. Stopping compute retains EBS storage charges and does not accept F6/G2.
Check actual process records, job.json, report and hashes before declaring completion.

After verified final backup, STOP the instance and review termination_protection=false before
terminating it. Verify the instance is terminated and its root EBS removed. Preserve S3 archives.
Never terminate an instance with unverified-only local checkpoints. No automatic termination.
Private state/tfvars and transport receipts stay outside Git. See docs/ec2_m2_f6.md for commands.
