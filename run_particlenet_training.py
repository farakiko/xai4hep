import argparse
from particlenet import (
    ParticleNet,
    TopTaggingDataset,
    make_file_loaders,
    save_model,
    load_model,
    training_loop
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_geometric.data import Data, Dataset
import time
import matplotlib.pyplot as plt
import matplotlib
import torch_geometric
import pandas as pd
import h5py
from torch_geometric.data import Data, DataListLoader, Batch
from torch_geometric.loader import DataLoader

import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import torch
import numpy as np

import json
import os
import os.path as osp

import matplotlib
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

parser.add_argument("--num_workers", type=int, default=2, help="number of subprocesses used for data loading")
parser.add_argument("--prefetch_factor", type=int, default=1, help="number of samples loaded in advance by each worker")
parser.add_argument("--outpath", type=str, default="./experiments/", help="output folder")
parser.add_argument("--model_prefix", type=str, default="ParticleNet_model", help="directory to hold the model and plots")
parser.add_argument("--dataset", type=str, default="./data/toptagging/", help="dataset path")
parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrites the model if True")
parser.add_argument("--n_epochs", type=int, default=3, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--patience", type=int, default=10, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--nearest", type=int, default=4, help="k nearest neighbors in gravnet layer")
parser.add_argument("--make_predictions", dest="make_predictions",
                    action="store_true", help="run inference on the test data")
parser.add_argument("--make_plots", dest="make_plots", action="store_true", help="makes plots of the test predictions")

args = parser.parse_args()


def setup(rank, world_size):
    """
    Necessary setup function that sets up environment variables and initializes the process group
    to perform training & inference using DistributedDataParallel (DDP). DDP relies on c10d ProcessGroup
    for communications, hence, applications must create ProcessGroup instances before constructing DDP.

    Args:
    rank: the process id (or equivalently the gpu index)
    world_size: number of gpus available
    """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # should be faster for DistributedDataParallel on gpus


def cleanup():
    """
    Necessary function that destroys the spawned process group at the end.
    """

    dist.destroy_process_group()


def run_demo(demo_fn, world_size, args, data_train, data_valid, model, num_classes, outpath):
    """
    Necessary function that spawns a process group of size=world_size processes to run demo_fn()
    on each gpu device that will be indexed by 'rank'.

    Args:
    demo_fn: function you wish to run on each gpu
    world_size: number of gpus available
    mode: 'train' or 'inference'
    """

    # mp.set_start_method('forkserver')

    mp.spawn(
        demo_fn,
        args=(world_size, args, data_train, data_valid, model, num_classes, outpath),
        nprocs=world_size,
        join=True,
    )


def train_ddp(rank, world_size, args, data_train, data_valid, model, num_classes, outpath):
    """
    A train_ddp() function that will be passed as a demo_fn to run_demo() to
    perform training over multiple gpus using DDP.

    It divides and distributes the training dataset appropriately, copies the model,
    wraps the model with DDP on each device to allow synching of gradients,
    and finally, invokes the training_loop() to run synchronized training among devices.
    """

    setup(rank, world_size)

    print(f"Running training on rank {rank}: {torch.cuda.get_device_name(rank)}")

    print('Building dataloaders...')
    data_train = data_train[int(rank * (len(data_train) / world_size)):int((rank + 1) * (len(data_train) / world_size))]
    data_valid = data_valid[int(rank * (len(data_valid) / world_size)):int((rank + 1) * (len(data_valid) / world_size))]

    train_loader = DataLoader(data_train, batch_size=args.batch_size)
    valid_loader = DataLoader(data_valid, batch_size=args.batch_size)

    # give each gpu a subset of the data
    hyper_train = int(len(dataset_train) / world_size)
    hyper_valid = int(len(dataset_valid) / world_size)

    # copy the model to the GPU with id=rank
    print(f"Copying the model on rank {rank}..")
    model = model.to(rank)
    model.train()
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)

    training_loop(
        rank,
        ddp_model,
        train_loader,
        valid_loader,
        args.batch_size,
        args.n_epochs,
        args.patience,
        optimizer,
        num_classes,
        outpath,
    )

    cleanup()


def train(device, world_size, args, data_train, data_valid, model, num_classes, outpath):
    """
    A train() function that will load the training dataset and start a training_loop
    on a single device (cuda or cpu).
    """

    if device == "cpu":
        print("Running training on cpu")
    else:
        print(f"Running training on: {torch.cuda.get_device_name(device)}")
        device = device.index

    print('Building dataloaders...')
    train_loader = DataLoader(data_train, batch_size=args.batch_size)
    valid_loader = DataLoader(data_valid, batch_size=args.batch_size)

    # move the model to the device (cuda or cpu)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    training_loop(
        device,
        model,
        train_loader,
        valid_loader,
        args.batch_size,
        args.n_epochs,
        args.patience,
        optimizer,
        num_classes,
        outpath,
    )


if __name__ == "__main__":

    world_size = torch.cuda.device_count()

    torch.backends.cudnn.benchmark = True

    # setup the input/output dimension of the model
    num_features = 7  # we have 7 features
    num_classes = 1  # we have 6 classes/pids for delphes

    outpath = osp.join(args.outpath, args.model_prefix)

    model_kwargs = {
        "node_feat_size": num_features,
        "num_classes": num_classes,
        "k": args.nearest,
    }

    model = ParticleNet(**model_kwargs)

    # save model_kwargs and hyperparameters
    save_model(args, args.model_prefix, outpath, model_kwargs)

    print(model)
    print(args.model_prefix)

    print("Training over {} epochs".format(args.n_epochs))

    # run the training using DDP if more than one gpu is available
    print('Loading training datafiles...')
    data_train = []
    for i in range(12):
        data_train = data_train + torch.load(f"{args.dataset}/train/processed/data_{i}.pt")
        print(f"- loaded file {i} for train")

    print('Loading validation datafiles...')
    data_valid = []
    for i in range(4):
        data_valid = data_valid + torch.load(f"{args.dataset}/val/processed/data_{i}.pt")
        print(f"- loaded file {i} for valid")

    if world_size >= 2:
        run_demo(train_ddp, world_size, args, data_train, data_valid, model, num_classes, outpath)
    else:
        train(device, world_size, args, data_train, data_valid, model, num_classes, outpath)
