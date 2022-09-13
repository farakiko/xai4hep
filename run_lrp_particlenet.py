import argparse
from models import ParticleNet
from explainer import LRP_ParticleNet
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
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import warnings
warnings.filterwarnings("ignore")


# this script runs lrp on a trained ParticleNet model

parser = argparse.ArgumentParser()

parser.add_argument("--outpath", type=str, default='./experiments/', help="path to the trained model directory")
parser.add_argument("--model", type=str, default="ParticleNet_model", help="Which model to load")
parser.add_argument("--data", type=str, default='./data/toptagging/train/processed/data_0.pt', help="path to datafile")

args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g. to run on prp
    python -u run_lrp_particlenet.py --model='ParticleNet_model' --outpath='/xai4hep/experiments/' --data='/xai4hep/toptagging/test/data_0.pt

    """

    # Check if the GPU configuration and define the global base device
    if torch.cuda.device_count() > 0:
        print(f'Will use {torch.cuda.device_count()} gpu(s)')
        print("GPU model:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:0')
    else:
        print('Will use cpu')
        device = torch.device('cpu')

    loader = DataLoader(torch.load(f"{args.data}"), batch_size=1, shuffle=True)

    # load a pretrained model and update the outpath
    with open(f"experiments/ParticleNet_model/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(f"{args.outpath}/{args.model}/best_epoch_weights.pth", map_location=device)

    model = ParticleNet(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)

    # run lrp
    batch_x_list, batch_y_list, Rscores_list, R_edges_list, edge_index_list = [], [], [], [], []

    for i, batch in enumerate(loader):
        # batch = Batch(x=batch2.x[:8], y=batch2.y[:8], batch=batch2.batch[:8])

        print(f'Explaining jet # {i}')
        print(f'Testing lrp on: \n {batch}')

        # run lrp on sample model
        model.eval()
        lrp = LRP_ParticleNet(device='cpu', model=model, epsilon=1e-8)
        Rscores, R_edges, edge_index = lrp.explain(batch, neuron_to_explain=0)

        batch_x_list.append(batch.x)
        batch_y_list.append(batch.y)
        Rscores_list.append(Rscores)
        R_edges_list.append(R_edges)
        edge_index_list.append(edge_index)

        print('------------------------------------------------------')
        if i == 2000:
            break

    # store the Rscores in the binder folder for further notebook plotting
    with open('binder/batch_x.pkl', 'wb') as handle:
        pkl.dump(batch_x_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('binder/batch_y.pkl', 'wb') as handle:
        pkl.dump(batch_y_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('binder/Rscores.pkl', 'wb') as handle:
        pkl.dump(Rscores_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('binder/R_edges.pkl', 'wb') as handle:
        pkl.dump(R_edges_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open('binder/edge_index.pkl', 'wb') as handle:
        pkl.dump(edge_index_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
