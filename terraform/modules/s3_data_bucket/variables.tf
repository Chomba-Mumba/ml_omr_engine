variable "bucket_name" {}
variable "env" {}
variable "lifecycle_rules" {}
variable "tags" {
    type    = map(string)
    default = {}
}
variable "enable_versioning" {
    default = false
}