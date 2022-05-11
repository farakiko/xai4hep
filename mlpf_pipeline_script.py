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
from plot_utils import make_Rmaps

# this script runs lrp on a trained MLPF model

parser = argparse.ArgumentParser()

parser.add_argument("--loader",         type=str,           default='junk/test_loader.pth',   help="path to a saved pytorch DataLoader")
parser.add_argument("--outpath",        type=str,           default='./data/test_tmp_delphes/experiments/',  help="path to the trained model directory")
parser.add_argument("--load_model",     type=str,           default="",     help="Which model to load")
parser.add_argument("--load_epoch",     type=int,           default=0,      help="Which epoch of the model to load")
parser.add_argument("--out_neuron",     type=int,           default=0,      help="the output neuron you wish to explain")
parser.add_argument("--pid",            type=str,           default="chhadron",     help="Which model to load")
parser.add_argument("--run_lrp",        dest='run_lrp',     action='store_true', help="runs lrp")
parser.add_argument("--make_rmaps",     dest='make_rmaps',  action='store_true', help="makes rmaps")
parser.add_argument("--size",           type=int,           default=0,      help="batch the events to fit in memory")

args = parser.parse_args()


def load_model(device, outpath, model_directory, load_epoch):
    PATH = outpath + '/epoch_' + str(load_epoch) + '_weights.pth'

    print('Loading a previously trained model..')
    with open(outpath + '/model_kwargs.pkl', 'rb') as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(PATH, map_location=device)

    if "DataParallel" in model_directory:   # if the model was trained using DataParallel then we do this
        state_dict = torch.load(PATH, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        state_dict = new_state_dict

    return state_dict, model_kwargs, outpath


if __name__ == "__main__":
    """
    e.g. to run lrp and make Rmaps
    python -u mlpf_pipeline_script.py --run_lrp --make_rmaps --load_model='MLPF_gen_ntrain_1_nepochs_1_clf_reg' --load_epoch=0 --outpath='/particleflowvol/test_tmp_delphes/experiments/' --loader='/particleflowvol/loader/test_loader.pth' --pid='chhadron' --size=400

    e.g. to only make Rmaps
    python -u mlpf_pipeline_script.py --make_rmaps --load_model='MLPF_gen_ntrain_1_nepochs_1_clf_reg' --load_epoch=0 --outpath='/particleflowvol/test_tmp_delphes/experiments/' --loader='/particleflowvol/loader/test_loader.pth' --pid='chhadron' --size=400
    """

    if args.run_lrp:
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

        # load a pretrained model and update the outpath
        outpath = args.outpath + args.load_model
        state_dict, model_kwargs, outpath = load_model(device, outpath, args.load_model, args.load_epoch)
        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)
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

        with open(f'{outpath}/Rtensors_list.pkl', 'wb') as f:
            pkl.dump(Rtensors_list, f)
        with open(f'{outpath}/inputs_list.pkl', 'wb') as f:
            pkl.dump(inputs_list, f)
        with open(f'{outpath}/preds_list.pkl', 'wb') as f:
            pkl.dump(preds_list, f)

    if args.make_rmaps:
        outpath = args.outpath + args.load_model
        with open(f'{outpath}/Rtensors_list.pkl',  'rb') as f:
            Rtensors_list = pkl.load(f)
        with open(f'{outpath}/inputs_list.pkl',  'rb') as f:
            inputs_list = pkl.load(f)
        with open(f'{outpath}/preds_list.pkl',  'rb') as f:
            preds_list = pkl.load(f)

        print('Making Rmaps..')
        make_Rmaps(args.outpath, Rtensors_list, inputs_list, preds_list, pid=args.pid, neighbors=3, out_neuron=args.out_neuron)
