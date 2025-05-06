terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.49.0"
    }
  }
  required_version = ">= 1.1.0"
 
  backend "s3" {
    bucket                  = "tf-awesome-backend"
    key                     = "terraform.tfstate"
    workspace_key_prefix    = "workspaces"
    region                  = "${var.region}"
    profile                 = "tf-awesome"
  }
}