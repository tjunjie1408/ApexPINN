terraform {
  required_version = ">= 1.6, < 2.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
}

provider "aws" {
  region              = "ap-southeast-2"
  profile             = "apexpinn"
  allowed_account_ids = ["985882743884"]
  default_tags { tags = { Project = "ApexPINN", ManagedBy = "Terraform", Purpose = "M2-F6-EC2" } }
}

variable "launch_instance" { default = false }
variable "subnet_id" { default = "subnet-073494a5f88add74f" }
variable "control_key" {
  type = string
  validation {
    condition     = can(regex("^inputs/[a-f0-9]{64}/apexpinn_ec2_control_v1.zip$", var.control_key))
    error_message = "Use the verified hash-addressed EC2 control archive key."
  }
}
variable "control_sha256" {
  type = string
  validation {
    condition     = can(regex("^[a-f0-9]{64}$", var.control_sha256))
    error_message = "Use the verified control ZIP SHA256."
  }
}
variable "termination_protection" { default = true }

data "aws_ami" "gpu" {
  owners = ["898082745236"]
  filter {
    name   = "image-id"
    values = ["ami-085f952c300f89090"]
  }
}

resource "aws_security_group" "gpu" {
  name_prefix = "apexpinn-m2-f6-ec2-"
  description = "No inbound ports; SSM and package/S3 downloads only"
  vpc_id      = "vpc-0cdc6a8095201e932"
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_iam_role" "gpu" {
  name = "apexpinn-m2-f6-ec2"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.gpu.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "s3" {
  role = aws_iam_role.gpu.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetBucketLocation"], Resource = "arn:aws:s3:::apexpinn-m2-f6-985882743884-ap-southeast-2" },
      {
        Effect    = "Allow", Action = ["s3:ListBucket"], Resource = "arn:aws:s3:::apexpinn-m2-f6-985882743884-ap-southeast-2"
        Condition = { StringLike = { "s3:prefix" = ["inputs/*", "ec2-runs/m2-f6-mid-v1/*"] } }
      },
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "arn:aws:s3:::apexpinn-m2-f6-985882743884-ap-southeast-2/inputs/*" },
      {
        Effect   = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"]
        Resource = "arn:aws:s3:::apexpinn-m2-f6-985882743884-ap-southeast-2/ec2-runs/m2-f6-mid-v1/*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "gpu" {
  name = "apexpinn-m2-f6-ec2"
  role = aws_iam_role.gpu.name
}

resource "aws_instance" "gpu" {
  count                                = var.launch_instance ? 1 : 0
  ami                                  = data.aws_ami.gpu.id
  instance_type                        = "g6.2xlarge"
  subnet_id                            = var.subnet_id
  associate_public_ip_address          = true
  vpc_security_group_ids               = [aws_security_group.gpu.id]
  iam_instance_profile                 = aws_iam_instance_profile.gpu.name
  disable_api_termination              = var.termination_protection
  instance_initiated_shutdown_behavior = "stop"
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  root_block_device {
    volume_size           = 100
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
    tags                  = { Name = "apexpinn-m2-f6-persistent-root" }
  }
  user_data = templatefile("${path.module}/bootstrap.sh.tftpl", {
    control_key = var.control_key, control_sha256 = var.control_sha256
  })
  user_data_replace_on_change = true
  tags                        = { Name = "apexpinn-m2-f6-l4-v1" }
  depends_on                  = [aws_iam_role_policy.s3, aws_iam_role_policy_attachment.ssm]
}

output "handoff" {
  value = {
    instance_id     = try(aws_instance.gpu[0].id, null)
    region          = "ap-southeast-2"
    instance_type   = "g6.2xlarge"
    ami             = data.aws_ami.gpu.id
    role_arn        = aws_iam_role.gpu.arn
    setup_only      = true
    training_status = "NOT RUN"
  }
}
