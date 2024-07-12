#!/bin/bash
source .env

mkdir -p $WORKING_DIR/dataset
curl -L "https://universe.roboflow.com/ds/oOA9yUpuh9?key=$ROBOFLOW_KEY" > $WORKING_DIR/dataset/roboflow.zip; unzip $WORKING_DIR/dataset/roboflow.zip -d $WORKING_DIR/dataset/roboflow.zip
