# Infrastructure references

AWS is excluded from the current F6 route. The active preparation uses Kaggle; Colab remains the
existing notebook/Drive workflow. This directory retains reusable infrastructure recipes, not a claim
that cloud training is currently deployed or complete.

| Recipe | Purpose and recorded limitation |
| :--- | :--- |
| [EC2](ec2/README.md) | Prepared L4 instance with full-state backup; recorded Sydney capacity failures prevented training |
| [SageMaker Studio](sagemaker/README.md) | Notebook/S3 alternative; the recorded Studio deployment was retired |
| [Managed training job](sagemaker_managed_job/README.md) | Separate container-based execution route, not the active user selection |

Do not apply Terraform, recreate compute or start training simply to validate a README. Region/account
access, quota and capacity must be checked when the user explicitly selects a future deployment.
Terraform state, plans, credentials and operational receipts stay local and ignored. Preserve state
needed for existing resource management; deleting a local cache does not delete cloud resources.

If an AWS route is selected again, verify result backup before stopping/deleting compute and verify
storage cleanup separately. Closing a browser does not stop paid resources. See
[EC2 completion steps](../docs/ec2_m2_f6.md#finish-and-remove-compute).
