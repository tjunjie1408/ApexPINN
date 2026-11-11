output "handoff" {
  value = {
    account_id         = data.aws_caller_identity.current.account_id
    region             = var.region
    bucket_name        = aws_s3_bucket.artifacts.bucket
    execution_role_arn = aws_iam_role.training.arn
    repository_url     = aws_ecr_repository.training.repository_url
    repository_name    = aws_ecr_repository.training.name
  }
}
