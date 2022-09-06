import random
import argparse
import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import numpy as np
import mplhep as hep
import pandas as pd

import torch
import torch_geometric
from torch_geometric.nn import GravNetConv

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

from explainer import LRP_ParticleNet
from models import ParticleNet
import jetnet

import warnings

warnings.filterwarnings("ignore")

# from plot_utils import make_Rmaps

# this script runs lrp on a trained MLPF model

parser = argparse.ArgumentParser()

parser.add_argument("--loader", type=str, default="junk/test_loader.pth", help="path to a saved pytorch DataLoader")
parser.add_argument("--outpath", type=str, default="./experiments/", help="path to the trained model directory")
parser.add_argument("--load_model", type=str, default="", help="Which model to load")
parser.add_argument("--load_epoch", type=int, default=0, help="Which epoch of the model to load")
parser.add_argument("--out_neuron", type=int, default=0, help="the output neuron you wish to explain")
parser.add_argument("--pid", type=str, default="chhadron", help="Which model to load")
parser.add_argument("--run_lrp", dest="run_lrp", action="store_true", help="runs lrp")
parser.add_argument("--make_rmaps", dest="make_rmaps", action="store_true", help="makes rmaps")
parser.add_argument("--size", type=int, default=0, help="batch the events to fit in memory")

args = parser.parse_args()


# get sample dataset https://jetnet.readthedocs.io/en/latest/   # what's called hls4ml
"""
jets here are 1TeV
in the ParticleNet top paper has jets smaller (and delphes)

cons: 30 particles is a simplification (can go up to 150)
"""
print("Fetching the data..")
dataset_top = jetnet.datasets.JetNet(jet_type="t")  # y=1

dataset_gluon = jetnet.datasets.JetNet(jet_type="g")  # y=0
dataset_q = jetnet.datasets.JetNet(jet_type="q")  # y=0

data[1]
torch.tensor(1)

# load the dataset in a convenient pyg format
dataset_pyg = []
for data in dataset_gluon:
    x = data[0][data[0][:, 3] == 0.5][:, :3]  # skip the mask
    d_gluon = Data(x=x, y=torch.tensor(1))
    dataset_pyg.append(d_gluon)

for data in dataset_top:
    x = data[0][data[0][:, 3] == 0.5][:, :3]  # skip the mask
    d_top = Data(x=x, y=torch.tensor(0))
    dataset_pyg.append(d_top)

for data in dataset_q:
    x = data[0][data[0][:, 3] == 0.5][:, :3]  # skip the mask
    d_q = Data(x=x, y=torch.tensor(0))
    dataset_pyg.append(d_q)

random.shuffle(dataset_pyg)

loader = DataLoader(dataset_pyg, batch_size=1, shuffle=False)

# load a pretrained model and update the outpath
model = ParticleNet(node_feat_size=3)

for batch in loader:
    if len(batch.x) != 30:
        break

preds, _, _, _ = model(batch)
