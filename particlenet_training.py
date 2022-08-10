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

parser.add_argument("--loader",         type=str,           default='junk/test_loader.pth',   help="path to a saved pytorch DataLoader")
parser.add_argument("--outpath",        type=str,           default='./experiments/',  help="path to the trained model directory")
parser.add_argument("--load_model",     type=str,           default="",     help="Which model to load")
parser.add_argument("--load_epoch",     type=int,           default=0,      help="Which epoch of the model to load")
parser.add_argument("--out_neuron",     type=int,           default=0,      help="the output neuron you wish to explain")
parser.add_argument("--pid",            type=str,           default="chhadron",     help="Which model to load")
parser.add_argument("--run_lrp",        dest='run_lrp',     action='store_true', help="runs lrp")
parser.add_argument("--make_rmaps",     dest='make_rmaps',  action='store_true', help="makes rmaps")
parser.add_argument("--size",           type=int,           default=0,      help="batch the events to fit in memory")

args = parser.parse_args()


# get sample dataset
print('Fetching the data..')
dataset_gluon = jetnet.datasets.JetNet(jet_type='g')
dataset_top = jetnet.datasets.JetNet(jet_type='t')
dataset_q = jetnet.datasets.JetNet(jet_type='q')

# load the dataset in a convenient pyg format
dataset_pyg = []
for data in dataset_gluon:
    d = Data(x=data[0], y=data[1])
    dataset_pyg.append(d)

for data in dataset_top:
    d = Data(x=data[0], y=data[1])
    dataset_pyg.append(d)

for data in dataset_q:
    d = Data(x=data[0], y=data[1])
    dataset_pyg.append(d)


random.shuffle(dataset_pyg)


dataset_pyg[1700].y
loader = DataLoader(dataset_pyg, batch_size=1, shuffle=False)

# load a pretrained model and update the outpath
model = ParticleNet(node_feat_size=4)

next(iter(loader)).x
for batch in loader:
    break
preds, _, _, _ = model(batch)


preds

batch.y