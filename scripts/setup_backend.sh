#setup terraform backend for storing data

echo "setting up backend for terraform environment states"

cd ../terraform/bootstrap

terraform init

terraform fmt

terraform plan

terraform apply -auto-approve