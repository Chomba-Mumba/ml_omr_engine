variable "aws_region" {
    type = string
}

variable "ecr_repo_name" {
    type = string
    default = "ml_ecr_repo"
}

variable "image_tag" {
    type = string
    default = "latest"
}

variable "force_image_rebuild" {
    type = bool
    default =  false
}

variable "aws_acc_id" {
    type = string
    sensitive = true
}
