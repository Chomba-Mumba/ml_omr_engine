resource "aws_s3_bucket" "s3-bucket" {
    bucket = "${var.bucket_name}-${var.env}"
    enable_versioning = "${var.enable_versioning}"

    acl = "${var.acl}"

    tags = "${var.tags}"

}