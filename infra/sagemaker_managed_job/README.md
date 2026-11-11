# SageMaker infrastructure preparation

Prepared, NOT APPLIED. Uses the named AWS profile `apexpinn`, account `985882743884`, Sydney.
Terraform creates a private versioned encrypted S3 bucket, immutable-tag ECR repository,
CloudWatch log group and a scoped SageMaker execution role. It creates no notebook, Studio
domain, endpoint, GPU instance or training job. A managed job is launched separately.

Run `terraform init`, then `terraform plan -out=setup.tfplan`. Review the exact permissions,
resource names and costs before running `terraform apply setup.tfplan`. Do not apply as part
of validation. Applying changes IAM access and creates persistent resources.

To prepare a remote container builder, use
`terraform plan -var=enable_remote_builder=true -out=setup.tfplan` instead. This adds a scoped
CodeBuild role/project and log group, but starts no build. CodeBuild builds are paid CPU work
and require a separate explicit user launch. Its privileged Docker mode is limited to that
build project. Do not use the training role to build images or expose Docker on the workstation.

The existing operator needs provisioning rights plus `iam:PassRole` for the exact execution
role, conditioned on `iam:PassedToService=sagemaker.amazonaws.com`, to submit training. The
training role cannot create resources, launch other jobs, change IAM or read arbitrary buckets.
The single `ecr:GetAuthorizationToken` wildcard is required by that AWS API; repository operations
are scoped. Input objects are read-only to training; run objects permit checkpoint sync writes
and deletes. S3 versioning preserves earlier versions, but does not make the prefix immutable.

If `/aws/sagemaker/TrainingJobs` already exists, import it before applying:
`terraform import aws_cloudwatch_log_group.training /aws/sagemaker/TrainingJobs`.
This resource is shared by SageMaker training jobs in the account/region; review its retention
setting before adoption. S3/ECR have `prevent_destroy` guards and are never force-deleted.
No lifecycle expiration deletes research checkpoints. Manage eventual cleanup separately.

State files and plan files may contain account details and are locally excluded by this
directory's `.gitignore`. Keep the generated handoff's Terraform state local too. Do not publish
state, `infra.json`, credentials or plan files. No credential is embedded in the configuration.

After applying, `terraform output -json handoff` returns the control-plane inputs. The launcher
does not require a Studio domain; open its notebook in any Python control environment with
boto3 with CRT, which is required for `aws login` credentials. See the handoff README for exact
transfer/build steps. Local fits are always forbidden.
