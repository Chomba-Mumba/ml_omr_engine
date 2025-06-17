#!/bin/bash

for dir in ./* do 
    if [ ! -d "$directory" ]; then
        continue
    fi
    for file in ./* do
        if [ "${file}" = "dockerfile" ] then
            docker build -t "${directory}:latest" .
            docker push "${ECR_URL}"
        fi
done