import argparse
import json
import os
import os.path as osp
import pickle as pkl
import sys
import time
import warnings
from glob import glob

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from explainer import LRP_ParticleNet
from models import ParticleNet
from particlenet import load_data

warnings.filterwarnings("ignore")

# Check if the GPU configuration and define the global base device
if torch.cuda.device_count() > 0:
    print(f"Will use {torch.cuda.device_count()} gpu(s)")
    print("GPU model:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    print("Will use cpu")
    device = torch.device("cpu")

# this script runs lrp on a trained ParticleNet model

parser = argparse.ArgumentParser()

parser.add_argument(
    "--outpath",
    type=str,
    default="./binder/",
    help="path to the trained model directory",
)
parser.add_argument(
    "--model_prefix",
    type=str,
    default="ParticleNet_dropout1",
    help="Which model to load",
)
parser.add_argument(
    "--dataset", type=str, default="./data/toptagging/", help="path to datafile"
)
parser.add_argument(
    "--model",
    type=int,
    default=-1,
    help="model to run Rscores for... -1=trained, x=untrained # x",
)
parser.add_argument("--quick", dest="quick", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g. to run on prp
    python -u run_lrp_particlenet.py --outpath='/xai4hepvol/' --dataset='/xai4hepvol/toptagging/' --model=-1

    """

    outpath = osp.join(args.outpath, args.model_prefix)

    # load the testing data
    print("- loading datafiles for lrp studies...")
    data_test = load_data(args.dataset, "test", 4, args.quick)
    loader = DataLoader(data_test, batch_size=1, shuffle=True)

    # load a pretrained model
    with open(f"{outpath}/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    if args.model == -1:  # load the best model
        state_dict = torch.load(
            f"{outpath}/weights/best_epoch_weights.pth", map_location=device
        )
    else:
        state_dict = torch.load(
            f"{outpath}/weights/before_training_weights_{args.model}.pth",
            map_location=device,
        )

    # the following line will make it possible to retrieve intermediate activations
    model_kwargs["for_LRP"] = True

    model = ParticleNet(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(model)

    # make directory to hold the Rscores
    if args.model == -1:
        PATH = f"{outpath}/Rscores_best/"
    else:
        PATH = f"{outpath}/Rscores_{args.model}/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # initilize an lrp class
    lrp = LRP_ParticleNet(device=device, model=model, epsilon=1e-8)

    # initilize placeholders
    R_edges_list, edge_index_list = [], []

    # to store the input and target for plotting purposes
    batch_x_list, batch_y_list = [], []
    # to store the p4 for plotting purposes
    batch_px_list, batch_py_list, batch_pz_list, batch_E_list = [], [], [], []

    ti = time.time()
    for i, jet in enumerate(loader):

        if i == 100:
            if args.quick:
                break

        if i == 1000:
            break

        print(f"Explaining jet # {i}: {jet}")

        # explain a single jet
        try:
            R_edges, edge_index = lrp.explain(jet.to(device))
        except:
            print("jet is not processed correctly so skipping it")
            continue

        batch_x_list.append(jet.x.detach().cpu())
        batch_y_list.append(jet.y.detach().cpu())

        # for fast jet
        batch_px_list.append(jet.px.detach().cpu())
        batch_py_list.append(jet.py.detach().cpu())
        batch_pz_list.append(jet.pz.detach().cpu())
        batch_E_list.append(jet.E.detach().cpu())

        R_edges_list.append(R_edges)
        edge_index_list.append(edge_index)

        print("------------------------------------------------------")

    tf = time.time()

    with open(f"{PATH}/time.json", "w") as fp:  # dump time
        json.dump(
            {
                "time_min": round((tf - ti) / 60, 3),
                "time_hours": round((tf - ti) / 3600, 3),
            },
            fp,
        )

    # store the Rscores in the binder folder for further notebook plotting
    with open(f"{PATH}/batch_x.pkl", "wb") as handle:
        pkl.dump(batch_x_list, handle)
    with open(f"{PATH}/batch_y.pkl", "wb") as handle:
        pkl.dump(batch_y_list, handle)

    # for fastjet
    with open(f"{PATH}/batch_px.pkl", "wb") as handle:
        pkl.dump(batch_px_list, handle)
    with open(f"{PATH}/batch_py.pkl", "wb") as handle:
        pkl.dump(batch_py_list, handle)
    with open(f"{PATH}/batch_pz.pkl", "wb") as handle:
        pkl.dump(batch_pz_list, handle)
    with open(f"{PATH}/batch_E.pkl", "wb") as handle:
        pkl.dump(batch_E_list, handle)

    with open(f"{PATH}/R_edges.pkl", "wb") as handle:
        pkl.dump(R_edges_list, handle)
    with open(f"{PATH}/edge_index.pkl", "wb") as handle:
        pkl.dump(edge_index_list, handle)
