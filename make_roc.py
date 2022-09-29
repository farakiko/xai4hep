import argparse
from models import ParticleNet
from particlenet import (
    TopTaggingDataset,
    make_file_loaders,
    save_model,
    load_model,
    training_loop
)

import os
import os.path as osp
import sys
from glob import glob
import time

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, DataListLoader, Batch
from torch_geometric.loader import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from sklearn.metrics import roc_curve, auc

import pickle as pkl
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hist as hist2

import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# define the global base device
if torch.cuda.device_count():
    device = torch.device("cuda:0")
else:
    device = "cpu"

"""
Developing a PyTorch Geometric ParticleNet training/inference pipeline using DistributedDataParallel.

Author: Farouk Mokhtar
"""

parser = argparse.ArgumentParser()

parser.add_argument("--outpath", type=str, default="./experiments/", help="output folder")
parser.add_argument("--model_prefix", type=str, default="ParticleNet_6", help="directory to hold the model and plots")
parser.add_argument("--dataset", type=str, default="./data/toptagging/", help="dataset path")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--wow", dest='wow', action='store_true')

args = parser.parse_args()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()

    torch.backends.cudnn.benchmark = True

    # setup the input/output dimension of the model
    num_features = 7  # we have 7 input features
    num_classes = 1  # we have one output node

    outpath = osp.join(args.outpath, args.model_prefix)

    # load the trained model
    with open(f"{outpath}/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(f"{outpath}/best_epoch_weights.pth", map_location=device)

    model = ParticleNet(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if args.wow:
        model.train()

    print(model)

    # quick test
    print('Loading testing datafiles...')
    data_test = []
    for i in range(4):
        data_test = data_test + torch.load(f"{args.dataset}/test/processed/data_{i}.pt")
        print(f"- loaded file {i} for test")
    loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

    sig = nn.Sigmoid()

    y_score = None
    y_test = None
    for i, batch in enumerate(loader):
        print(f"making prediction on sample # {i}")
        # if i == 3:
        #     break
        preds, _, _, _ = model(batch.to(device))
        preds = sig(preds).detach().cpu()

        if y_score == None:
            y_score = preds[:].reshape(-1).cpu()
            y_test = batch.y.cpu()
        else:
            y_score = torch.cat([y_score.cpu(), preds[:].reshape(-1).cpu()])
            y_test = torch.cat([y_test.cpu(), batch.y.cpu()])

    # save the predictions
    print("saving the predictions")
    torch.save(y_test, f"{outpath}/y_test.pt")
    torch.save(y_score, f"{outpath}/y_score.pt")

    # Compute ROC curve and ROC area for each class

    from matplotlib.font_manager import FontProperties

    font = FontProperties()
    font.set_name('Times New Roman')
    plt.rcParams.update({'font.family': 'serif'})

    print("making the Roc curves")
    fpr, tpr, _ = roc_curve(y_test.cpu(), y_score.cpu())
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(
        tpr,
        fpr,
        color="darkorange",
        lw=2,
        label=f"AUC = {round(auc(fpr, tpr)*100,2)}%",
    )
    plt.xlim([0.0, 1.0])
    plt.ylabel("False Positive Rate", fontsize=25)
    plt.xlabel("True Positive Rate", fontsize=25)
    plt.yscale('log')
    plt.legend(loc="lower right", fontsize=25)
    plt.savefig(f"{outpath}/Roc_curve.pdf")
