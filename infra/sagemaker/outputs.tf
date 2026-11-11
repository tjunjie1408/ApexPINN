output "handoff" {
  value = {
    account_id        = data.aws_caller_identity.current.account_id
    region            = var.region
    bucket_name       = aws_s3_bucket.artifacts.bucket
    studio_domain_id  = aws_sagemaker_domain.studio.id
    user_profile_name = aws_sagemaker_user_profile.owner.user_profile_name
    space_name        = aws_sagemaker_space.training.space_name
    instance_type     = var.instance_type
    compute_started   = var.start_compute
  }
}

output "cleanup_reminder" {
  value = "After verified S3 backup: STOP/DELETE the Studio JupyterLab app and verify Deleted. Closing the browser does not stop billing. Delete space/EBS only after backup; preserve S3 results."
}
