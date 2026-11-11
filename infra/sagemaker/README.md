# Studio JupyterLab and S3

**Sydney deployment retired at the user's request on 2026-11-09.** Do not apply this historical
recipe against the retained Sydney state: it would recreate Studio. Only the inputs bucket
remains. The local S3-only retirement configuration and verified final state are under
`data/aws_setup/sydney_retirement_20261109/`. Both checked US destinations are currently denied
by organization policy; no replacement deployment exists. See the local RECORD.md for evidence.

This is the default Studio recipe. Actual deployment status is recorded locally in RECORD.md;
Terraform state and deployment receipts stay excluded from Git. The previous unapplied managed-job
recipe is preserved in `infra/sagemaker_managed_job`; do not apply both for the same task.
This default defines no ECR repository, CodeBuild project, custom image or managed training job.

Uses `apexpinn`, account985882743884, Sydney. Existing VPC/subnets were read from AWS on
2026-11-09 and are explicit defaults in variables.tf. Check them if using another account.
Studio uses PublicInternetOnly for package downloads, with an NFS security group allowing
only self-group inbound2049. No NAT gateway or public inbound SSH/Jupyter rule is created.

Terraform creates private/versioned/encrypted S3, a scoped Studio execution role, a domain,
user profile and private50GiB space. The built-in AWS SageMaker Distribution GPU image is used.
Default size is the user-selected **ml.g6.2xlarge**, one L4,8vCPUs,32GiB RAM. The reviewed
ml.g6.4xlarge option still has one L4, with more host CPU/RAM; its speed benefit is unmeasured.

`start_compute=false` is the default. Applying that plan creates domain/storage resources but
does NOT launch the GPU app. Domain EFS, space EBS and S3 can incur storage charges even when
compute is stopped. Storage is protected against accidental deletion; it is not cost-free.

```powershell
terraform init
terraform plan -out=studio.tfplan
# Review resource names, security permissions and costs before applying.
terraform apply studio.tfplan
terraform output -json handoff
```

Starting compute is a separate paid user action. Review
`terraform plan -var=start_compute=true -out=start.tfplan`, then apply only when you choose
to start. This starts the app, not training. Open the private space from Studio using your
user profile, upload the notebook/control folder and set up its uv environment. No Docker build.

The operator needs provisioning permissions and PassRole for this exact Studio execution role.
The notebook role can read this bucket's inputs and read/write its studio-runs prefix.
The separate portal policy reads this domain/profile/space/app and the profile's default
JupyterServer compatibility probe, lists Studio metadata in Sydney,
opens this profile, and lets the user configure this private space and start/stop its default
JupyterLab app. It cannot delete the space, create training jobs, edit IAM or delete S3 results.
Granting portal access does not start compute. Launching the app is still a paid user action.
AWS may create its SageMaker service-linked role during initial domain provisioning.

Idle shutdown is enabled with a120-minute default. Foreground notebook execution keeps the
kernel busy; confirm actual idle behavior during remote validation. Backgrounding a process
can leave the kernel idle and is not used by this runner. Idle shutdown is a fallback, not a
guaranteed account spending limit or permission to leave compute running after F6.

## Completion reminder: STOP and DELETE compute

After F6 results and S3 snapshot are verified, **stop the Studio JupyterLab app and verify its
status is Deleted**. DeleteApp stops/removes the app's underlying compute. Closing the notebook
or browser does not stop billing. This reminder is also a Terraform output and notebook section.

If Terraform started the app, review/apply a new plan with `start_compute=false` to remove it.
If the Console started an app outside Terraform, stop it in Studio or use this explicit command,
replacing DOMAIN with the output domain ID:

```powershell
aws sagemaker delete-app --domain-id DOMAIN --space-name m2-f6 --app-type JupyterLab --app-name default --profile apexpinn --region ap-southeast-2
aws sagemaker describe-app --domain-id DOMAIN --space-name m2-f6 --app-type JupyterLab --app-name default --profile apexpinn --region ap-southeast-2
```

These are instructions, NOT commands already executed. Do not stop an active training or backup
process without understanding which checkpoint is durable. Removing the app retains the space's
EBS data. Deleting the space removes that data and requires verified backup first; its
prevent_destroy guard must be deliberately reviewed before removal. Domain EFS is retained too.
Review EBS/EFS/S3 storage separately if cleaning up costs. Preserve scientific S3 archives.

Do not run a blanket `terraform destroy`: S3 and the space are deliberately protected. No
automatic app/space deletion is performed. No completion-triggered automation was scheduled;
the notebook shows the reminder and the agent will repeat it during F6 completion review.
