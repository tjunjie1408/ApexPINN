variable "profile" {
  type    = string
  default = "apexpinn"
}

variable "region" {
  type    = string
  default = "ap-southeast-2"
  validation {
    condition     = var.region == "ap-southeast-2"
    error_message = "This reviewed handoff targets Sydney only."
  }
}

variable "account_id" {
  type    = string
  default = "985882743884"
  validation {
    condition     = can(regex("^[0-9]{12}$", var.account_id))
    error_message = "An AWS account ID must have12 digits."
  }
}

variable "name" {
  type    = string
  default = "apexpinn-m2-f6"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,24}$", var.name))
    error_message = "Use a short lowercase name with letters, digits and hyphens."
  }
}

variable "vpc_id" {
  type    = string
  default = "vpc-0cdc6a8095201e932"
}

variable "subnet_ids" {
  type    = list(string)
  default = ["subnet-073494a5f88add74f", "subnet-004096a64f9cd3f59", "subnet-09d302ba64cb75151"]
}

variable "instance_type" {
  type    = string
  default = "ml.g6.2xlarge"
  validation {
    condition     = contains(["ml.g6.2xlarge", "ml.g6.4xlarge"], var.instance_type)
    error_message = "Use one of the reviewed single-L4 sizes. Keep the size fixed within a run."
  }
}

variable "start_compute" {
  description = "Starting an app incurs GPU charges. False prepares domain/space without launching compute."
  type        = bool
  default     = false
}

variable "user_profile_name" {
  type    = string
  default = "apexpinn"
}

variable "space_name" {
  type    = string
  default = "m2-f6"
}
