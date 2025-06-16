provider "aws" {
    region = $var.aws_region
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
    bucket = var.ml_tf_backend_bucket
    key = "./"
    region = var.aws_region
    encyrpt = true
    use_lockfile = true
  }
}
