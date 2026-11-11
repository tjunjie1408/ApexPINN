data "aws_caller_identity" "current" {}
data "aws_vpc" "studio" { id = var.vpc_id }

locals {
  bucket_name = "${var.name}-${var.account_id}-${var.region}"
  image_arn   = "arn:aws:sagemaker:${var.region}:648430277019:image/sagemaker-distribution-gpu"
}

resource "aws_s3_bucket" "artifacts" {
  bucket        = local.bucket_name
  force_destroy = false
  lifecycle { prevent_destroy = true }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_policy" "tls" {
  bucket = aws_s3_bucket.artifacts.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Deny", Principal = "*", Action = "s3:*"
      Resource  = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_security_group" "studio" {
  name_prefix = "${var.name}-studio-"
  description = "Studio NFS within the same security group; no public inbound access"
  vpc_id      = data.aws_vpc.studio.id
  ingress {
    description = "Domain EFS NFS within Studio"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    self        = true
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_iam_role" "studio" {
  name = "${var.name}-studio"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow", Principal = { Service = "sagemaker.amazonaws.com" }, Action = "sts:AssumeRole"
      Condition = {
        StringEquals = { "aws:SourceAccount" = var.account_id }
        ArnLike      = { "aws:SourceArn" = "arn:aws:sagemaker:${var.region}:${var.account_id}:*" }
      }
    }]
  })
}

resource "aws_iam_role_policy" "studio" {
  role = aws_iam_role.studio.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetBucketLocation"], Resource = aws_s3_bucket.artifacts.arn },
      {
        Effect    = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.artifacts.arn
        Condition = { StringLike = { "s3:prefix" = ["inputs/*", "studio-runs/*"] } }
      },
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.artifacts.arn}/inputs/*" },
      {
        Effect   = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"]
        Resource = "${aws_s3_bucket.artifacts.arn}/studio-runs/*"
      }
    ]
  })
}

resource "aws_sagemaker_domain" "studio" {
  domain_name             = var.name
  auth_mode               = "IAM"
  app_network_access_type = "PublicInternetOnly"
  vpc_id                  = data.aws_vpc.studio.id
  subnet_ids              = var.subnet_ids
  default_user_settings {
    execution_role      = aws_iam_role.studio.arn
    security_groups     = [aws_security_group.studio.id]
    studio_web_portal   = "ENABLED"
    default_landing_uri = "studio::"
    # AWS returns portal defaults that this provider otherwise proposes removing.
    studio_web_portal_settings {}
    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type       = var.instance_type
        sagemaker_image_arn = local.image_arn
      }
      app_lifecycle_management {
        idle_settings {
          lifecycle_management        = "ENABLED"
          idle_timeout_in_minutes     = 120
          min_idle_timeout_in_minutes = 60
          max_idle_timeout_in_minutes = 1440
        }
      }
    }
  }
  retention_policy { home_efs_file_system = "Retain" }
  depends_on = [aws_iam_role_policy.studio]
}

resource "aws_sagemaker_user_profile" "owner" {
  domain_id         = aws_sagemaker_domain.studio.id
  user_profile_name = var.user_profile_name
}

resource "aws_sagemaker_space" "training" {
  domain_id  = aws_sagemaker_domain.studio.id
  space_name = var.space_name
  ownership_settings { owner_user_profile_name = aws_sagemaker_user_profile.owner.user_profile_name }
  space_sharing_settings { sharing_type = "Private" }
  space_settings {
    app_type = "JupyterLab"
    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type                 = var.instance_type
        sagemaker_image_arn           = local.image_arn
        sagemaker_image_version_alias = "4.6.2"
      }
      app_lifecycle_management {
        idle_settings { idle_timeout_in_minutes = 120 }
      }
    }
    space_storage_settings {
      ebs_storage_settings { ebs_volume_size_in_gb = 50 }
    }
  }
  lifecycle { prevent_destroy = true }
}

resource "aws_sagemaker_app" "training" {
  count      = var.start_compute ? 1 : 0
  domain_id  = aws_sagemaker_domain.studio.id
  space_name = aws_sagemaker_space.training.space_name
  app_type   = "JupyterLab"
  app_name   = "default"
  resource_spec {
    instance_type       = var.instance_type
    sagemaker_image_arn = local.image_arn
  }
}
