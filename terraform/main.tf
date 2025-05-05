
module "s3_data_bucket" {
  source = "./modules/s3_data_bucket"
  #variables here
  tags = {
    environment = "dev"
    project = "ml_omr_engine"
    public-bucket = false
  }
}