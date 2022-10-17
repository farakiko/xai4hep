import json
import os
import os.path as osp
import pickle as pkl
import shutil
import sys
from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataListLoader, DataLoader


def save_model(args, outpath, model_kwargs, kernel_sizes, fc_size, dropout, depth):

    if not args.overwrite and os.path.isfile(
        f"{args.outpath}/{args.model_prefix}/best_epoch_weights.pth"
    ):
        print(f"model {args.model_prefix} already exists, please delete it")
        sys.exit(0)

    filelist = [
        f for f in os.listdir(outpath) if not f.endswith(".txt")
    ]  # don't remove the logs.txt

    for f in filelist:
        try:
            os.remove(os.path.join(outpath, f))
        except:
            shutil.rmtree(os.path.join(outpath, f))

    with open(f"{outpath}/model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(f"{outpath}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump(
            {
                "n_epochs": args.n_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "nearest": args.nearest,
                "kernel_sizes": kernel_sizes,
                "fc_size": fc_size,
                "dropout": dropout,
                "depth": depth,
            },
            fp,
        )


def load_data(dataset_path, flag, n_files, quick):

    if quick:  # use only 1000 events for train
        print(f"--- loading only 1000 events for quick {flag}")
        data = torch.load(f"{dataset_path}/{flag}/processed/data_0.pt")[:1000]
    else:
        data = []
        for i in range(n_files):
            data += torch.load(f"{dataset_path}/{flag}/processed/data_{i}.pt")
            print(f"--- loaded file {i} for {flag}")

    return data


def make_roc(y_test, y_score, path):
    fpr, tpr, _ = roc_curve(y_test, y_score)
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
    plt.ylabel("False Positive Rate")
    plt.xlabel("True Positive Rate")
    plt.yscale("log")
    plt.legend(loc="lower right")
    plt.savefig(path)
