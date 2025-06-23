
data "aws_iam_policy_document" "ml_ecr_policy" {
    statement {
        sid = "anyone can pull from repi"
        effect = "Allow"
        principals {
            type = "AWS"
            identifiers = ["*"]
        }
        actions = [
            "ecr:GetDownloadUrlForLayer",
            "ecr: BatchGetImage",
            "ecr:ListImages"
        ]
        condition {
            test = "StringEquals"
            variable = "aws:PrincipalAccount"
            values = ["${var.aws_acc_id}"]
        }
    }
    
    statement {
        sid = "allow push only from github actions"
        effect = "Allow"
        principals {
            type = "AWS"
            identifiers = ["arn:aws:iam::${var.aws_acc_id}:role/github_actions_omr_engine_role"]
        }
        actions = ["ecr:BatchCheckLayerAvailability",
            "ecr:CompleteLayerUpload",
            "ecr:InitiateLayerUpload",
            "ecr:PutImage",
            "ecr:UploadLayerPart"]
            condition {
                test = "StringEquals"
                variable = "aws:PrincipalAccount"
                values = ["${var.aws_acc_id}"]
            }
    }
}