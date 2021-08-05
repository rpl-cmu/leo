#!/bin/bash

LOCAL_DIR="local/"
mkdir -p $LOCAL_DIR

wget https://www.dropbox.com/s/r1nabquau6zcxmc/datasets.zip -P $LOCAL_DIR

unzip $LOCAL_DIR/datasets.zip -d $LOCAL_DIR

rm -r $LOCAL_DIR/datasets.zip