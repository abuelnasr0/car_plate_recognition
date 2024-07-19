#!/bin/bash
source .env

mkdir -p $WORKING_DIR/models && cd $WORKING_DIR/models && mkdir -p pretrained
for var in "$@"
do
    curl -L  https://github.com/ultralytics/assets/releases/download/v8.2.0/$var > pretrained/$var; 
    echo "$var downloaded"
done
