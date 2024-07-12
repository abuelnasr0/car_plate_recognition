#!/bin/bash
source .env

mkdir -p $WORKING_DIR/models && cd $WORKING_DIR/models && mkdir -p pretrained
curl -L  https://github.com/ultralytics/assets/releases/download/v8.2.0/${1} > pretrained/${1}; 