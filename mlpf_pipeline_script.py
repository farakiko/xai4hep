import argparse
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
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

from explainer import LRP_MLPF
from models import MLPF

# this script runs lrp on a trained MLPF model

"""
e.g run locally as:
python3 -u mlpf_pipeline_script.py --model_weights='/particleflowvol/test_tmp_delphes/experiments/MLPF_gen_ntrain_1_nepochs_1_clf_reg/epoch_0_weights.pth' --model_kwargs='/particleflowvol/test_tmp_delphes/experiments/MLPF_gen_ntrain_1_nepochs_1_clf_reg/model_kwargs.pkl' --outpath='/particleflowvol/test_tmp_delphes/experiments/MLPF_gen_ntrain_1_nepochs_1_clf_reg/' --size=400
"""

parser = argparse.ArgumentParser()

# for saving the model
parser.add_argument("--loader",         type=str,           default='./test_loader.pth',                                 help="path to a saved pytorch DataLoader")
parser.add_argument("--model_weights",  type=str,           default="./data/test_tmp_delphes/experiments/MLPF_gen_ntrain_1_nepochs_1_clf_reg/epoch_0_weights.pth",            help="path to a trained model's weights.pth")
parser.add_argument("--model_kwargs",   type=str,           default="./data/test_tmp_delphes/experiments/MLPF_gen_ntrain_1_nepochs_1_clf_reg/model_kwargs.pkl",            help="path to a trained model's model.kwargs.pkl")
parser.add_argument("--out_neuron",     type=int,           default=0,                                                   help="the output neuron you wish to explain")
parser.add_argument("--outpath",        type=str,           default='./',                                                help="path to save the Rtensors")
parser.add_argument("--size",           type=int,           default=100,      help="size of event")

args = parser.parse_args()


def load_model(args):

    with open(args.model_kwargs, 'rb') as f:
        model_kwargs = pkl.load(f)

    model = MLPF(**model_kwargs)

    state_dict = torch.load(args.model_weights, map_location=device)

    if "DataParallel" in args.model_weights:   # if the model was trained using DataParallel then we do this
        state_dict = torch.load(args.model_weights, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    # Check if the GPU configuration and define the global base device
    if torch.cuda.device_count() > 0:
        print(f'Will use {torch.cuda.device_count()} gpu(s)')
        print("GPU model:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:0')
    else:
        print('Will use cpu')
        device = torch.device('cpu')

    # get sample dataset
    print('Fetching the data..')
    loader = torch.load(args.loader)

    # load a pretrained model
    print('Loading a previously trained model..')
    model = load_model(args)
    model.to(device)

    # run lrp
    Rtensors_list, preds_list, inputs_list = [], [], []

    for i, event in enumerate(loader):
        print(f'Explaining event # {i}')

        # break it down to a smaller part for lrp (to avoid memory issues)
        def get_small_batch(event, size):
            small_batch = Batch()
            small_batch.x = event.x[:size]
            small_batch.ygen = event.ygen[:size]
            small_batch.ygen_id = event.ygen_id[:size]
            small_batch.ycand = event.ycand[:size]
            small_batch.ycand_id = event.ycand_id[:size]
            small_batch.batch = event.batch[:size]
            return small_batch

        event = get_small_batch(event, size=args.size)
        print(f'Testing lrp on: \n {event}')

        # run lrp on sample model
        model.eval()
        lrp_instance = LRP_MLPF(device, model, epsilon=1e-9)
        Rtensor, pred, input = lrp_instance.explain(event, neuron_to_explain=args.out_neuron)

        Rtensors_list.append(Rtensor.detach().to('cpu'))
        preds_list.append(pred.detach().to('cpu'))
        inputs_list.append(input.detach().to('cpu').to_dict())
        break

    with open(f'{args.outpath}/Rtensors_list.pkl', 'wb') as f:
        pkl.dump(Rtensors_list, f)
    with open(f'{args.outpath}/preds_list.pkl', 'wb') as f:
        pkl.dump(preds_list, f)
    with open(f'{args.outpath}/inputs_list.pkl', 'wb') as f:
        pkl.dump(inputs_list, f)
