#!/bin/bash

DATASET_NAME="nav2d"
DATASET_DIR="local/datasets/sim"

echo "Downloading dataset $DATASET_NAME to $DATASET_DIR"

mkdir -p $DATASET_DIR

wget https://www.dropbox.com/s/13thjbmh1ypmbcr/$DATASET_NAME.zip -P $DATASET_DIR
unzip $DATASET_DIR/$DATASET_NAME.zip -d $DATASET_DIR
rm -r $DATASET_DIR/$DATASET_NAME.zip