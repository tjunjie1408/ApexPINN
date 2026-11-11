variable "enable_remote_builder" {
  description = "Prepare an AWS CodeBuild image builder. Applying Terraform never starts a build."
  type        = bool
  default     = false
}

resource "aws_cloudwatch_log_group" "build" {
  count             = var.enable_remote_builder ? 1 : 0
  name              = "/aws/codebuild/${var.name}"
  retention_in_days = var.log_retention_days
}

resource "aws_iam_role" "build" {
  count = var.enable_remote_builder ? 1 : 0
  name  = "${var.name}-image-build"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow", Principal = { Service = "codebuild.amazonaws.com" }, Action = "sts:AssumeRole"
      Condition = {
        StringEquals = { "aws:SourceAccount" = var.account_id }
        ArnEquals    = { "aws:SourceArn" = "arn:aws:codebuild:${var.region}:${var.account_id}:project/${var.name}" }
      }
    }]
  })
}

resource "aws_iam_role_policy" "build" {
  count = var.enable_remote_builder ? 1 : 0
  role  = aws_iam_role.build[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetBucketLocation", "s3:GetBucketAcl"], Resource = aws_s3_bucket.artifacts.arn },
      { Effect = "Allow", Action = ["s3:GetObject", "s3:GetObjectVersion"], Resource = "${aws_s3_bucket.artifacts.arn}/build/*" },
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      {
        Effect   = "Allow"
        Action   = ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage"]
        Resource = aws_ecr_repository.training.arn
      },
      {
        Effect   = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = [aws_cloudwatch_log_group.build[0].arn, "${aws_cloudwatch_log_group.build[0].arn}:*"]
      }
    ]
  })
}

resource "aws_codebuild_project" "image" {
  count          = var.enable_remote_builder ? 1 : 0
  name           = var.name
  service_role   = aws_iam_role.build[0].arn
  build_timeout  = 30
  queued_timeout = 30
  artifacts { type = "NO_ARTIFACTS" }
  environment {
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true
    environment_variable {
      name  = "REPOSITORY_URI"
      value = aws_ecr_repository.training.repository_url
    }
    environment_variable {
      name  = "IMAGE_TAG"
      value = "set-a-versioned-tag-before-starting"
    }
  }
  source {
    type     = "S3"
    location = "${aws_s3_bucket.artifacts.bucket}/build/source-location-must-be-overridden.zip"
    buildspec = yamlencode({
      version = "0.2"
      phases = {
        pre_build = {
          commands = ["aws ecr get-login-password --region \"$AWS_DEFAULT_REGION\" | docker login --username AWS --password-stdin \"$${REPOSITORY_URI%%/*}\""]
        }
        build = { commands = ["docker build --platform linux/amd64 -t \"$REPOSITORY_URI:$IMAGE_TAG\" ."] }
        post_build = {
          commands = ["test \"$CODEBUILD_BUILD_SUCCEEDING\" -eq 1", "docker push \"$REPOSITORY_URI:$IMAGE_TAG\""]
        }
      }
    })
  }
  logs_config {
    cloudwatch_logs {
      group_name = aws_cloudwatch_log_group.build[0].name
      status     = "ENABLED"
    }
  }
  depends_on = [aws_iam_role_policy.build]
}

output "image_builder_name" {
  value = var.enable_remote_builder ? aws_codebuild_project.image[0].name : null
}
