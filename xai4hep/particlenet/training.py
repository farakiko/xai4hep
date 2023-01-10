import json
import math
import os
import os.path as osp
import pickle as pkl
import sys
import time
from glob import glob
from typing import Callable, Optional, Union

import h5py
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch.nn import Linear
from torch_cluster import knn_graph
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


@torch.no_grad()
def validation_run(multi_gpu, device, model, loader):
    with torch.no_grad():
        optimizer = None
        ret = train(
            multi_gpu,
            device,
            model,
            loader,
            optimizer,
        )
    return ret


def train(multi_gpu, device, model, loader, optimizer):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

    criterion = nn.BCELoss()
    sig = nn.Sigmoid()

    # initialize loss and time counters
    losses, t = 0, 0

    for batch in loader:

        if multi_gpu:
            batch = batch
        else:
            batch = batch.to(device)

        # run forward pass
        t0 = time.time()
        preds, targets = model(batch)
        t1 = time.time()
        t += t1 - t0

        loss = criterion(sig(preds), targets.reshape(-1, 1).float())

        # backprop
        if is_train:  # not run during a validation run
            for param in model.parameters():
                # better than calling optimizer.zero_grad()
                # according to https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                param.grad = None
            loss.backward()
            optimizer.step()

        losses = losses + loss.detach()

    print(
        f"Average inference time per batch is {round((t / len(loader)), 3)}s"
    )

    losses = (losses / (len(loader))).cpu().item()

    return losses


def training_loop(
    multi_gpu,
    device,
    model,
    train_loader,
    valid_loader,
    n_epochs,
    patience,
    optimizer,
    outpath,
):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        model: a pytorch model wrapped by DistributedDataParallel (DDP)
        train_loader: a pytorch Dataloader that loads .pt files for training when you invoke the get() method
        valid_loader: a pytorch Dataloader that loads .pt files for validation when you invoke the get() method
        patience: number of stale epochs allowed before stopping the training
        optimizer: optimizer to use for training (by default: Adam)
        outpath: path to store the model weights and training plots
    """

    # create directory to hold loss plots
    if not os.path.exists(f"{outpath}/loss_plots/"):
        os.makedirs(f"{outpath}/loss_plots/")

    # create directory to hold the model state at each epoch
    if not os.path.exists(f"{outpath}/epoch_weights/"):
        os.makedirs(f"{outpath}/epoch_weights/")

    t0_initial = time.time()

    losses_train, losses_valid = [], []

    best_val_loss = 99999.9
    stale_epochs = 0

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        print(f"---->Initiating a training run")
        model.train()
        losses = train(
            multi_gpu,
            device,
            model,
            train_loader,
            optimizer,
        )

        losses_train.append(losses)

        # validation step
        print(f"---->Initiating a validation run")
        model.eval()
        losses = validation_run(multi_gpu, device, model, valid_loader)

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

            with open(
                f"{outpath}/best_epoch.json", "w"
            ) as fp:  # dump best epoch
                json.dump({"best_epoch": epoch}, fp)
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = n_epochs - (epoch + 1)
        time_per_epoch = (t1 - t0_initial) / (epoch + 1)
        eta = epochs_remaining * time_per_epoch / 60

        print(
            f"epoch={epoch + 1} / {n_epochs} "
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
        torch.save(
            state_dict, f"{outpath}/epoch_weights/epoch_{epoch+1}_weights.pth"
        )

        # make loss plots
        fig, ax = plt.subplots()
        ax.plot(range(len(losses_train)), losses_train, label="training")
        ax.plot(range(len(losses_valid)), losses_valid, label="validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        plt.savefig(f"{outpath}/loss_plots/losses_epoch_{epoch}.pdf")
        plt.close(fig)

        with open(f"{outpath}/loss_plots/losses_epoch_{epoch}.pkl", "wb") as f:
            pkl.dump((losses_train, losses_valid), f)

        print("----------------------------------------------------------")
    print(
        f"Done with training. Total training time is {round((time.time() - t0_initial)/60,3)}min"
    )
