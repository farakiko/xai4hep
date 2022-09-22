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
parser.add_argument("--model", type=str, default="ParticleNet_6", help="Which model to load")
parser.add_argument("--data", type=str, default='./data/toptagging/test/processed/data_0.pt', help="path to datafile")

args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g. to run on prp
    python -u run_lrp_particlenet.py --model='ParticleNet_model' --outpath='/xai4hep/experiments/' --data='/xai4hep/toptagging/test/data_0.pt'

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
    with open(f"{args.outpath}/{args.model}/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    # state_dict = torch.load(f"{args.outpath}/{args.model}/best_epoch_weights.pth", map_location=device)
    # state_dict = torch.load(f"{args.outpath}/{args.model}/epoch_0_weights.pth", map_location=device)
    state_dict = torch.load(f"{args.outpath}/{args.model}/before_training_weights.pth", map_location=device)

    model = ParticleNet(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(model)

    # run lrp
    lrp = LRP_ParticleNet(device='cpu', model=model, epsilon=1e-8)
    batch_x_list, batch_y_list, Rscores_list, R_edges_list, edge_index_list = [], [], [], [], []
    batch_px_list, batch_py_list, batch_pz_list, batch_E_list = [], [], [], []

    for i, jet in enumerate(loader):

        if i == 1000:
            break

        print(f'Explaining jet # {i}')
        print(f'Testing lrp on: \n {jet}')

        # explain jet
        try:
            Rscores, R_edges, edge_index, R_scores_b4 = lrp.explain(jet, neuron_to_explain=0)
        except:
            print('jet is broken so skipping')
            continue

        # check the accuracy of the input Rscores
        rtol = 3
        if torch.abs(Rscores.sum() - R_scores_b4.sum()) > rtol:
            print('--> High Input Rscore error due to the depth of the model')

        batch_x_list.append(jet.x)
        batch_y_list.append(jet.y)

        # for fast jet
        batch_px_list.append(jet.px)
        batch_py_list.append(jet.py)
        batch_pz_list.append(jet.pz)
        batch_E_list.append(jet.E)

        # Rscores_list.append(Rscores)
        R_edges_list.append(R_edges['edge_conv_2'])
        edge_index_list.append(edge_index['edge_conv_2'])

        print('------------------------------------------------------')

    # store the Rscores in the binder folder for further notebook plotting
    with open(f'binder/{args.model}/Rscores/batch_x.pkl', 'wb') as handle:
        pkl.dump(batch_x_list, handle)
    with open(f'binder/{args.model}/Rscores/batch_y.pkl', 'wb') as handle:
        pkl.dump(batch_y_list, handle)

    # for fastjet
    with open(f'binder/{args.model}/Rscores/batch_px.pkl', 'wb') as handle:
        pkl.dump(batch_px_list, handle)
    with open(f'binder/{args.model}/Rscores/batch_py.pkl', 'wb') as handle:
        pkl.dump(batch_py_list, handle)
    with open(f'binder/{args.model}/Rscores/batch_pz.pkl', 'wb') as handle:
        pkl.dump(batch_pz_list, handle)
    with open(f'binder/{args.model}/Rscores/batch_E.pkl', 'wb') as handle:
        pkl.dump(batch_E_list, handle)

    # with open(f'binder/{args.model}/Rscores/Rscores.pkl', 'wb') as handle:
    #     pkl.dump(Rscores_list, handle)
    with open(f'binder/{args.model}/Rscores/R_edges.pkl', 'wb') as handle:
        pkl.dump(R_edges_list, handle)
    with open(f'binder/{args.model}/Rscores/edge_index.pkl', 'wb') as handle:
        pkl.dump(edge_index_list, handle)
