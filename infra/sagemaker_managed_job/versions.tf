terraform {
  required_version = ">= 1.6, < 2.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region              = var.region
  profile             = var.profile
  allowed_account_ids = [var.account_id]
  default_tags {
    tags = { Project = "ApexPINN", ManagedBy = "Terraform", Purpose = "M2-F6" }
  }
}
