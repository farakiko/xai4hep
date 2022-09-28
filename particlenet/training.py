import sklearn.metrics
import sklearn
from torch_geometric.data import Data, Dataset
import time
import matplotlib.pyplot as plt
import matplotlib
import torch_geometric
from torch_geometric.loader import DataListLoader, DataLoader
import pandas as pd
import h5py
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from typing import Callable, Optional, Union
from torch_geometric.data import Data, DataListLoader, Batch
from torch_geometric.loader import DataLoader

import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_cluster import knn_graph

import numpy as np

import json
import math
import os
import time

import matplotlib
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams.update({'font.size': 20})

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


@torch.no_grad()
def validation_run(rank, model, train_loader, valid_loader, batch_size, num_classes, outpath):
    with torch.no_grad():
        optimizer = None
        ret = train(rank, model, train_loader, valid_loader, batch_size, optimizer, num_classes, outpath)
    return ret


def train(rank, model, train_loader, valid_loader, batch_size, optimizer, num_classes, outpath):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

    if is_train:
        print(f"---->Initiating a training run on rank {rank}")
        model.train()
        loader = train_loader
    else:
        print(f"---->Initiating a validation run rank {rank}")
        model.eval()
        loader = valid_loader

    criterion = nn.BCELoss()
    sig = nn.Sigmoid()

    # initialize loss counters
    losses = 0

    t0, tf = time.time(), 0
    t = 0
    for i, batch in enumerate(loader):

        # transformations for better training
        batch.x[:, 2] = (batch.x[:, 2] - 1.7) * 0.7  # part_pt_log
        batch.x[:, 3] = (batch.x[:, 3] - 2.0) * 0.7  # part_e_log
        batch.x[:, 4] = (batch.x[:, 4] + 4.7) * 0.7  # part_logptrel
        batch.x[:, 5] = (batch.x[:, 5] + 4.7) * 0.7  # part_logerel
        batch.x[:, 6] = (batch.x[:, 6] - 0.2) * 4.7  # part_deltaR

        # run forward pass
        t0 = time.time()
        preds, _, _, _ = model(batch.to(rank))
        t1 = time.time()
        t = t + (t1 - t0)

        loss = criterion(sig(preds), batch.y.reshape(-1, 1).float())

        # backprop
        if is_train:
            for param in model.parameters():
                # better than calling optimizer.zero_grad()
                # according to https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                param.grad = None
            loss.backward()
            optimizer.step()

        losses = losses + loss.detach()

        # if i == 2:
        #     break

    print(f"Average inference time per batch on rank {rank} is {round((t / len(loader)), 3)}s")

    t0 = time.time()

    losses = (losses / (len(loader))).cpu().item()

    return losses


def training_loop(
    rank,
    model,
    train_loader,
    valid_loader,
    batch_size,
    n_epochs,
    patience,
    optimizer,
    num_classes,
    outpath,
):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        rank: int representing the gpu device id, or str=='cpu' (both work, trust me)
        model: a pytorch model wrapped by DistributedDataParallel (DDP)
        train_loader: a pytorch Dataloader that loads .pt files for training when you invoke the get() method
        valid_loader: a pytorch Dataloader that loads .pt files for validation when you invoke the get() method
        patience: number of stale epochs allowed before stopping the training
        optimizer: optimizer to use for training (by default: Adam)
        num_classes: number of particle candidate classes to predict (6 for delphes, 9 for cms)
        outpath: path to store the model weights and training plots
    """

    # create directory to hold training plots
    if not os.path.exists(outpath + "/training_plots/"):
        os.makedirs(outpath + "/training_plots/")
    if not os.path.exists(f"{outpath}/training_plots/losses/"):
        os.makedirs(f"{outpath}/training_plots/losses/")

    t0_initial = time.time()

    losses_train, losses_valid = [], []

    best_val_loss = 99999.9
    stale_epochs = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        model.train()

        # if epoch <= 8:
        #     lr = (3 + 3.375 * epoch) * 1e-4
        # elif epoch <= 16:
        #     lr = 3e-3 - (3.375 * (epoch - 8)) * 1e-4
        # elif epoch <= 20:
        #     lr = 3e-4 - (0.7487 * (epoch - 16)) * 1e-4
        # elif epoch <= 24:
        #     lr = 5e-7

        # optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

        losses = train(
            rank, model, train_loader, valid_loader, batch_size, optimizer, num_classes, outpath
        )

        losses_train.append(losses)

        # validation step
        model.eval()
        losses = validation_run(
            rank, model, train_loader, valid_loader, batch_size, num_classes, outpath
        )

        losses_valid.append(losses)

        # early-stopping
        if losses < best_val_loss:
            best_val_loss = losses
            stale_epochs = 0

            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()
            torch.save(state_dict, f"{outpath}/best_epoch_weights.pth")

            with open(f"{outpath}/best_epoch.json", "w") as fp:  # dump best epoch
                json.dump({"best_epoch": epoch}, fp)
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = n_epochs - (epoch + 1)
        time_per_epoch = (t1 - t0_initial) / (epoch + 1)
        eta = epochs_remaining * time_per_epoch / 60

        print(
            f"Rank {rank}: epoch={epoch + 1} / {n_epochs} "
            + f"train_loss={round(losses_train[epoch], 4)} "
            + f"valid_loss={round(losses_valid[epoch], 4)} "
            + f"stale={stale_epochs} "
            + f"time={round((t1-t0)/60, 2)}m "
            + f"eta={round(eta, 1)}m"
        )

        # save the model's weights
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, f"{outpath}/epoch_{epoch}_weights.pth")

        # make loss plots
        fig, ax = plt.subplots()
        ax.plot(range(len(losses_train)), losses_train, label="training")
        ax.plot(range(len(losses_valid)), losses_valid, label="validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        plt.savefig(f"{outpath}/training_plots/losses/loss_{epoch}.pdf")
        plt.close(fig)

        with open(f"{outpath}/training_plots/losses/loss_{epoch}_train.pkl", "wb") as f:
            pkl.dump(losses_train, f)
        with open(f"{outpath}/training_plots/losses/loss_{epoch}_valid.pkl", "wb") as f:
            pkl.dump(losses_valid, f)

        print("----------------------------------------------------------")
    print(f"Done with training. Total training time on rank {rank} is {round((time.time() - t0_initial)/60,3)}min")
