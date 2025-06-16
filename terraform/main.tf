provider "aws" {
    region = $var.aws_region
}

terraform {
  backend "s3" {
    bucket = var.ml_tf_backend_bucket
    key = "./"
    region = var.aws_region
    encyrpt = true
    use_lockfile = true
  }
  
  required_version = ">= 0.12"
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}
