import argparse
import json
import os
import os.path as osp
import pickle as pkl
import sys
import time
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataListLoader, DataLoader

sys.path.insert(0, "..")
from models import ParticleNet

from particlenet import load_data, make_roc

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 20})

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# define the global base device
world_size = torch.cuda.device_count()
multi_gpu = world_size >= 2
if world_size:
    device = torch.device("cuda:0")
    for i in range(world_size):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"
    print("Device: CPU")

# define argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--outpath", type=str, default="../experiments/", help="output folder"
)
parser.add_argument(
    "--model_prefix",
    type=str,
    default="ParticleNet_model",
    help="directory to hold the model and plots",
)
parser.add_argument(
    "--dataset", type=str, default="../data/toptagging/", help="dataset path"
)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument(
    "--quick", dest="quick", action="store_true", help="make inference on small sample"
)

args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g.
    python run_inference.py --quick --model_prefix='ParticleNet_model' --dataset="/xai4hepvol/toptagging/" --outpath="/xai4hepvol/experiments/"

    """

    outpath = osp.join(args.outpath, args.model_prefix)

    # load the best trained model for testing
    with open(f"{outpath}/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(f"{outpath}/best_epoch_weights.pth", map_location=device)

    model = ParticleNet(**model_kwargs)
    model.load_state_dict(state_dict)

    print("- loading datafiles for testing...")
    data_test = load_data(args.dataset, "test", 4, args.quick)

    if multi_gpu:
        test_loader = DataListLoader(
            data_test, batch_size=args.batch_size, shuffle=True
        )
        model = torch_geometric.nn.DataParallel(model)
    else:
        test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

    model.to(device)
    model.eval()

    print("- making predictions")
    y_score = None
    y_test = None
    for i, batch in enumerate(test_loader):

        if multi_gpu:
            batch = batch
        else:
            batch = batch.to(device)

        preds, targets = model(batch)
        preds = preds.detach().cpu()

        if y_score == None:
            y_score = preds[:].detach().cpu().reshape(-1)
            y_test = targets.detach().cpu()
        else:
            y_score = torch.cat([y_score, preds[:].detach().cpu().reshape(-1)])
            y_test = torch.cat([y_test, targets.detach().cpu()])

    # save the predictions
    print("- saving predictions")
    torch.save(y_test, f"{outpath}/y_test.pt")
    torch.save(y_score, f"{outpath}/y_score.pt")

    # Compute ROC curve
    print("- making Roc curves")
    make_roc(y_test, y_score, f"{outpath}/Roc_curve.pdf")
