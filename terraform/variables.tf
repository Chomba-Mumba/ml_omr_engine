variable "aws_region" {
    type = string
}

variable "ecr_repo_name" {
    type = string
    default = "ml_ecr_repo"
}

variable "aws_acc_id" {
    type = string
    sensitive = true
}
