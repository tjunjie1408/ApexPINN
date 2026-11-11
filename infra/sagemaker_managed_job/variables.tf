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

variable "log_retention_days" {
  type    = number
  default = 30
}
