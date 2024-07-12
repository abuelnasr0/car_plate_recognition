#!/bin/bash
source .env

mkdir -p $WORKING_DIR/dataset
curl -L "https://universe.roboflow.com/ds/jnQmbRCmgf?key=$ROBOFLOW_KEY" > $WORKING_DIR/dataset/roboflow.zip; unzip $WORKING_DIR/dataset/roboflow.zip -d $WORKING_DIR/dataset/roboflow.zip
