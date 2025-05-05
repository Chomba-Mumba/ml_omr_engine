output "s3_arn" {
    value = aws_s3_bucket.s3-bucket.s3_arn
    description = "S3 bucket ID."
}