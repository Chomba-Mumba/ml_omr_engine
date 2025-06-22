#!/bin/bash

for dir in ./* do 
    #if not a directory then skip
    if [ ! -d "$directory" ]; then
        continue
    fi

    #check for dockerfile and build
    for file in ./* do
        if [ "${file}" = "dockerfile" ] then
            docker build -t "${directory}:latest" .
            docker push "${ECR_REPOSITORY}"
        fi
done