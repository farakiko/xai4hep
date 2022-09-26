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

parser.add_argument("--outpath", type=str, default='./binder/', help="path to the trained model directory")
parser.add_argument("--model_prefix", type=str, default="ParticleNet_3", help="Which model to load")
parser.add_argument("--dataset", type=str, default='./data/toptagging/', help="path to datafile")
parser.add_argument("--epoch", type=int, default=-1, help="which epoch to run Rscores on")

args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g. to run on prp
    python -u scaling_lrp_particlenet.py --epoch=-1 --model='ParticleNet_3' --outpath='/xai4hepvol/' --dataset='/xai4hepvol/toptagging/'

    """

    # Check if the GPU configuration and define the global base device
    if torch.cuda.device_count() > 0:
        print(f'Will use {torch.cuda.device_count()} gpu(s)')
        print("GPU model:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:0')
    else:
        print('Will use cpu')
        device = torch.device('cpu')

    outpath = osp.join(args.outpath, args.model_prefix)

    # quick test
    print('Loading testing datafiles...')
    data_test = []
    for i in range(1):
        data_test = data_test + torch.load(f"{args.dataset}/test/processed/data_{i}.pt")
        print(f"- loaded file {i} for test")
    loader = DataLoader(data_test, batch_size=1, shuffle=True)

    # load a pretrained model and update the outpath
    with open(f"{outpath}/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    if args.epoch == -1:
        state_dict = torch.load(f"{outpath}/weights/best_epoch_weights.pth", map_location=device)
    else:
        state_dict = torch.load(
            f"{outpath}/weights/before_training_weights_{args.epoch}.pth", map_location=device)

    model = ParticleNet(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(model)

    # run lrp
    lrp = LRP_ParticleNet(device=device, model=model, epsilon=1e-8)
    batch_x_list, batch_y_list, Rscores_list, R_edges_list, edge_index_list = [], [], [], [], []
    batch_px_list, batch_py_list, batch_pz_list, batch_E_list = [], [], [], []

    for i, jet in enumerate(loader):

        if i == 10:
            break

        print(f'Explaining jet # {i}')
        print(f'Testing lrp on: \n {jet}')

        # explain jet
        # try:
        R_edges, edge_index = lrp.explain(jet, neuron_to_explain=0)
        # except:
        #     print("jet is not processed correctly so skipping it")
        #     continue

        batch_x_list.append(jet.x.detach().cpu())
        batch_y_list.append(jet.y.detach().cpu())

        # for fast jet
        batch_px_list.append(jet.px.detach().cpu())
        batch_py_list.append(jet.py.detach().cpu())
        batch_pz_list.append(jet.pz.detach().cpu())
        batch_E_list.append(jet.E.detach().cpu())

        R_edges_list.append(R_edges)
        edge_index_list.append(edge_index)

        print('------------------------------------------------------')

    if args.epoch == -1:
        PATH = f'{outpath}/Rscores_best/'
    else:
        PATH = f'{outpath}/Rscores_{args.epoch}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # store the Rscores in the binder folder for further notebook plotting
    with open(f'{PATH}/batch_x.pkl', 'wb') as handle:
        pkl.dump(batch_x_list, handle)
    with open(f'{PATH}/batch_y.pkl', 'wb') as handle:
        pkl.dump(batch_y_list, handle)

    # for fastjet
    with open(f'{PATH}/batch_px.pkl', 'wb') as handle:
        pkl.dump(batch_px_list, handle)
    with open(f'{PATH}/batch_py.pkl', 'wb') as handle:
        pkl.dump(batch_py_list, handle)
    with open(f'{PATH}/batch_pz.pkl', 'wb') as handle:
        pkl.dump(batch_pz_list, handle)
    with open(f'{PATH}/batch_E.pkl', 'wb') as handle:
        pkl.dump(batch_E_list, handle)

    with open(f'{PATH}/R_edges.pkl', 'wb') as handle:
        pkl.dump(R_edges_list, handle)
    with open(f'{PATH}/edge_index.pkl', 'wb') as handle:
        pkl.dump(edge_index_list, handle)
