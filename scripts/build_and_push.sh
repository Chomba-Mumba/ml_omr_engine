#!/bin/bash

for dir in ./* do 
    #if not a directory then skip
    if [ ! -d "$directory" ]; then
        continue
    fi

    #check for dockerfile and build
    for file in ./* do
        if [ "${file}" = "dockerfile" ] then
            docker build -t "${directory}:${IMAGE_TAG}" .
            docker push "${directory}:${IMAGE_TAG}4"
        fi
done