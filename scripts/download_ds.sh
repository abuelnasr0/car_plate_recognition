#!/bin/bash
source .env

mkdir -p $WORKING_DIR/dataset2
curl -L "https://universe.roboflow.com/ds/jnQmbRCmgf?key=$ROBOFLOW_KEY" > $WORKING_DIR/dataset2/roboflow.zip; unzip $WORKING_DIR/dataset2/roboflow.zip -d $WORKING_DIR/dataset2/roboflow.zip
