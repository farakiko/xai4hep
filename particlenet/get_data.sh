#!/bin/bash

set -e

# make data/ directory to hold the cms/ directory of datafiles under particleflow/
mkdir -p ../data/toptagging/
cd ../data/toptagging/

# get the data from https://zenodo.org/record/2603256#.Yxhl3OxByrO
mkdir -p train/raw/
cd train/raw/
wget https://zenodo.org/record/2603256/files/train.h5?download=1
mv train* train.h5
cd ../
mkdir processed/
cd ../

mkdir -p val/raw/
cd val/raw/
wget https://zenodo.org/record/2603256/files/val.h5?download=1
mv val* val.h5
cd ../
mkdir processed/
cd ../

mkdir -p test/raw/
cd test/raw/
wget https://zenodo.org/record/2603256/files/test.h5?download=1
mv test* test.h5
cd ../
mkdir processed/
cd ../

# process the datasets
cd ../../particlenet/
python TopTaggingDataset.py --dataset ../data/toptagging/ --mode train
python TopTaggingDataset.py --dataset ../data/toptagging/ --mode val
python TopTaggingDataset.py --dataset ../data/toptagging/ --mode test
