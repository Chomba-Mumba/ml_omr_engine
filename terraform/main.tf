provider "aws" {
    region = var.aws_region
}

terraform {
  required_version = "~> 1.12"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.100.0"
    }
  }

  backend "s3" {
    bucket = ""
    key = ""
    region = ""
    encrypt = true
    use_lockfile = true
  }
}
