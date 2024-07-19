#!/bin/bash
source .env

mkdir -p $WORKING_DIR/models && cd $WORKING_DIR/models && mkdir -p pretrained
for var in "$@"
do
    printf "\n"
    echo "Downloading $var"  
    printf "\n"
    curl -L  https://github.com/ultralytics/assets/releases/download/v8.2.0/$var > pretrained/$var; 
    printf "\n"
    echo "$var downloaded"
    printf "\n"
done
