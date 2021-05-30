#!/bin/bash

mkdir data
mkdir data/person

wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt
mv ConfLongDemo_JSI.txt data/person/

wget --no-check-certificate https://gin.g-node.org/v-goncharenko/neiry-demons/raw/master/nery_demons_dataset.zip
unzip nery_demons_dataset.zip -d data/demons
