#!/bin/bash

for dir in ./*; do 
    #if not a directory then skip
    if [ ! -d "$dir" ]; then
        continue
    fi

    #check for dockerfile and build
    for file in ./*; do
        if [ "${file}" = "dockerfile" ]; then
            docker build -t "${dir}:${IMAGE_TAG}" "$dir"
            docker push "${dir}:${IMAGE_TAG}"
        fi
    done
done