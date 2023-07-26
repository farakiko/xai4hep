#!/bin/bash

set -e

python dataset.py --dataset ../data/toptagging/ --mode train
python dataset.py --dataset ../data/toptagging/ --mode val
python dataset.py --dataset ../data/toptagging/ --mode test
