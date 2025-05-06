# This environment is for the setup of the dev, staging, prod and setup environment as their backends are hosted on s3

#required permissions: S3: ListBucket, S3:GetObject, S3:PutObject
resource "aws_iam_policy" "policy" {
  name        = "ml-omr-backend-s3-policy"
  path        = "/"
  description = "Policy for accessing the state bucket store on S3."

  policy = jsonencode({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": "arn:aws:s3:::mybucket",
                "Condition": {
                    "StringEquals": {
                    "s3:prefix": "${var.state_bucket}/envs/*"
                    }
                }
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": [
                    "arn:aws:s3:::${var.state_bucket}/envs/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": [
                    "arn:aws:s3:::${var.state_bucket}/envs/*"
                ]
            }
        ]
    })
}

resource aws_s3_bucket "state_bucket" {
    bucket = "${var.state_bucket}"

    tags = {
        project = "ml_omr_engine"
        public-bucket = false
    }
}

