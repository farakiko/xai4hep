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
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataListLoader, DataLoader


def save_model(
    args, model_fname, outpath, model_kwargs, kernel_sizes, fc_size, dropout, depth
):

    if not osp.isdir(outpath):
        os.makedirs(outpath)

    else:  # if directory already exists
        if not args.overwrite:  # if not overwrite then exit
            print(f"model {model_fname} already exists, please delete it")
            sys.exit(0)

        print(f"model {model_fname} already exists, deleting it")

        filelist = [
            f for f in os.listdir(outpath) if not f.endswith(".txt")
        ]  # don't remove the newly created logs.txt
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


def load_model(device, outpath, model_directory, load_epoch):
    if load_epoch == -1:
        PATH = outpath + "/best_epoch_weights.pth"
    else:
        PATH = outpath + "/epoch_" + str(load_epoch) + "_weights.pth"

    print("Loading a previously trained model..")
    with open(outpath + "/model_kwargs.pkl", "rb") as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(PATH, map_location=device)

    # # if the model was trained using DataParallel then we do this
    # state_dict = torch.load(PATH, map_location=device)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove module.
    #     new_state_dict[name] = v
    # state_dict = new_state_dict

    return state_dict, model_kwargs, outpath


class Collater:
    """
    This function was copied from torch_geometric.loader.Dataloader() source code.
    Edits were made such that the function can collate samples as a list of tuples
    of Data() objects instead of Batch() objects. This is needed becase pyg Dataloaders
    do not handle num_workers>0 since Batch() objects cannot be directly serialized using pkl.
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return batch

        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")


def make_file_loaders(
    world_size, dataset, num_files=1, num_workers=0, prefetch_factor=2
):
    """
    This function is only one line, but it's worth explaining why it's needed
    and what it's doing. It uses native torch Dataloaders with a custom collate_fn
    that allows loading Data() objects from pt files in a fast way. This is needed
    becase pyg Dataloaders do not handle num_workers>0 since Batch() objects
    cannot be directly serialized using pkl.

    Args:
        world_size: number of gpus available
        dataset: custom dataset
        num_files: number of files to load with a single get() call
        num_workers: number of workers to use for fetching files
        prefetch_factor: number of files to fetch in advance

    Returns:
        a torch iterable() that returns a list of 100 elements,
        each element is a tuple of size=num_files containing Data() objects
    """
    if world_size > 0:
        return torch.utils.data.DataLoader(
            dataset,
            num_files,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=Collater(),
            pin_memory=True,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            num_files,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=Collater(),
            pin_memory=False,
        )
