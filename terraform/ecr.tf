resource "aws_ecr_repository" "ml_ecr_repo"{
    name = var.ecr_repo_name
    image_tag_mutability = "MUTABLE"
    image_scanning_configuration {
        scan_on_push = true
    }

    encryption_configuration {
      encryption_type = "KMS"
    }
}

resource "aws_ecr_lifecycle_policy" "ml_ecr_lifecycle_policy" {
  repository = aws_ecr_repository.ml_ecr_repo.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description   = "Keep only 5 images"
        selection     = {
          countType        = "imageCountMoreThan"
          countNumber      = 5
          tagStatus        = "tagged"
          tagPrefixList   = ["prod"]
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_registry_scanning_configuration" "scan_configuration" {
  scan_type = "ENHANCED"

  rule {
    scan_frequency = "SCAN_ON_PUSH"
    repository_filter {
      filter = "*"
      filter_type = "WILDCARD"
    }
  }
}