import argparse
import json
import os
import os.path as osp
import pickle as pkl
import sys
import time
from glob import glob

import h5py
import hist as hist2
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch_geometric
from sklearn.metrics import auc, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch, Data, DataListLoader
from torch_geometric.loader import DataLoader

sys.path.insert(0, "..")
from models import ParticleNet

from particlenet import (
    TopTaggingDataset,
    load_model,
    make_file_loaders,
    save_model,
    training_loop,
)

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})

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

parser.add_argument(
    "--outpath", type=str, default="./experiments/", help="output folder"
)
parser.add_argument(
    "--model_prefix",
    type=str,
    default="ParticleNet_depth2",
    help="directory to hold the model and plots",
)
parser.add_argument(
    "--dataset", type=str, default="./data/toptagging/", help="dataset path"
)
parser.add_argument("--batch_size", type=int, default=100)

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
    # model.eval()

    # quick test
    print("Loading testing datafiles...")
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
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1])
    plt.ylabel("False Positive Rate")
    plt.xlabel("True Positive Rate")
    plt.yscale("log")
    # plt.title("")
    plt.legend(loc="lower right")
    plt.savefig(f"{outpath}/Roc_curve.pdf")
    plt.savefig(f"{outpath}/Roc_curve_1.pdf")
    plt.savefig(f"{outpath}/Roc_curve_10.pdf")