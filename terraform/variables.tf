variable "aws_region" {
    type = string
}

variable "ecr_repo_name" {
    type = string
    default = "ml_ecr_repo"
}

variable "ecr_registry" {
    type = string
    default = "${aws_acc}.dkr.ecr.${aws_region}.amazonaws.com"
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
