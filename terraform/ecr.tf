
resource "aws_ecr_repository" "ml_ecr_repo"{
    name = var.ecr_repo_name
    image_tag_mutability = "MUTABLE"
    image_scanning_configuration {
        scan_on_push = true
    }
}

resource "null_resource" "build_push_dkr_img" {
    triggers  {
        detect_docker_source_changes = var.force_iamge_rebuild == true ? timestamp()
    }
    provisioner "local_exec"{
        command = local.dkr_build_cmd
    }
}

resource "aws_ecr_lifecycle_policy" "ml_ecr_policy" {
  repository = aws_ecr_repository.ml_ecr_repo.name

  policy = jsonencode({
    rules = [
      {
        rule_priority = 1
        description   = "Keep only 5 images"
        selection     = {
          count_type        = "imageCountMoreThan"
          count_number      = 5
          tag_status        = "tagged"
          tag_prefix_list   = ["prod"]
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}