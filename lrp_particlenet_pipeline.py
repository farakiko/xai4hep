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

from explainer import LRP_ParticleNet
from models import ParticleNet
import jetnet

import warnings
warnings.filterwarnings("ignore")

# from plot_utils import make_Rmaps

# this script runs lrp on a trained MLPF model

parser = argparse.ArgumentParser()

parser.add_argument("--loader",         type=str,           default='junk/test_loader.pth',   help="path to a saved pytorch DataLoader")
parser.add_argument("--outpath",        type=str,           default='./experiments/',  help="path to the trained model directory")
parser.add_argument("--load_model",     type=str,           default="",     help="Which model to load")
parser.add_argument("--load_epoch",     type=int,           default=0,      help="Which epoch of the model to load")
parser.add_argument("--out_neuron",     type=int,           default=0,      help="the output neuron you wish to explain")
parser.add_argument("--pid",            type=str,           default="chhadron",     help="Which model to load")
parser.add_argument("--run_lrp",        dest='run_lrp',     action='store_true', help="runs lrp")
parser.add_argument("--make_rmaps",     dest='make_rmaps',  action='store_true', help="makes rmaps")
parser.add_argument("--size",           type=int,           default=0,      help="batch the events to fit in memory")

args = parser.parse_args()


if __name__ == "__main__":
    """
    e.g. to run lrp and make Rmaps
    python -u lrp_particlenet_pipeline.py --run_lrp --make_rmaps --load_model='MLPF_gen_ntrain_1_nepochs_1_clf_reg' --load_epoch=0 --outpath='/particleflowvol/test_tmp_delphes/experiments/' --loader='/particleflowvol/loader/test_loader.pth' --pid='chhadron' --size=400

    e.g. to only make Rmaps
    python -u lrp_particlenet_pipeline.py --make_rmaps --load_model='MLPF_gen_ntrain_1_nepochs_1_clf_reg' --load_epoch=0 --outpath='/particleflowvol/test_tmp_delphes/experiments/' --loader='/particleflowvol/loader/test_loader.pth' --pid='chhadron' --size=400
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
        dataset = jetnet.datasets.JetNet(jet_type='g')

        # load the dataset in a convenient pyg format
        dataset_pyg = []
        for data in dataset:
            d = Data(x=data[0], y=data[1])
            dataset_pyg.append(d)

        loader = DataLoader(dataset_pyg, batch_size=1, shuffle=False)

        # load a pretrained model and update the outpath
        model = ParticleNet(node_feat_size=4)
        # state_dict = torch.load('state_dict.pth', map_location=device)
        # model.load_state_dict(state_dict)
        # model.to(device)

        # run lrp
        Rscores_list, R_edges_list, edge_index_list = [], [], []

        for i, batch in enumerate(loader):
            print(f'Explaining jet # {i}')

            print(f'Testing lrp on: \n {batch}')

            # run lrp on sample model
            model.eval()
            lrp = LRP_ParticleNet(device='cpu', model=model, epsilon=1e-8)
            Rscores, R_edges, edge_index = lrp.explain(batch, neuron_to_explain=0)

            with open('binder/batch_x.pkl', 'wb') as handle:
                pkl.dump(batch.x, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('binder/Rscores.pkl', 'wb') as handle:
                pkl.dump(Rscores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('binder/R_edges.pkl', 'wb') as handle:
                pkl.dump(R_edges, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('binder/edge_index.pkl', 'wb') as handle:
                pkl.dump(edge_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

            Rscores_list.append(Rscores)
            R_edges_list.append(R_edges)
            edge_index_list.append(edge_index)

            break

        # with open(f'{outpath}/Rscores_list.pkl', 'wb') as f:
        #     pkl.dump(Rscores_list, f)
        # with open(f'{outpath}/R_edges_list.pkl', 'wb') as f:
        #     pkl.dump(R_edges_list, f)
        # with open(f'{outpath}/edge_index_list.pkl', 'wb') as f:
        #     pkl.dump(edge_index_list, f)
    #
    # if args.make_rmaps:
    #     outpath = args.outpath + args.load_model
    #     with open(f'{outpath}/Rtensors_list.pkl',  'rb') as f:
    #         Rtensors_list = pkl.load(f)
    #     with open(f'{outpath}/inputs_list.pkl',  'rb') as f:
    #         inputs_list = pkl.load(f)
    #     with open(f'{outpath}/preds_list.pkl',  'rb') as f:
    #         preds_list = pkl.load(f)
    #
    #     print('Making Rmaps..')
    #     make_Rmaps(args.outpath, Rtensors_list, inputs_list, preds_list, pid=args.pid, neighbors=3, out_neuron=args.out_neuron)
