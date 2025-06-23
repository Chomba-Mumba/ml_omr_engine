#!/bin/bash
cd ../

for dir in ./*; do 
    #if not a directory then skip
    if [ ! -d "$dir" ]; then
        continue
    fi

    #check for dockerfile and build
    if [ -f "$dir/Dockerfile" ]; then
        image_uri="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

        echo "building docker image ${dir}:${IMAGE_TAG}"
        docker build -t "$image_uri" "$dir"

        echo "pushing image to docker..."
        docker push "$image_uri"
    fi
done