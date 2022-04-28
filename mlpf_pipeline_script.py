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


size = 600


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
    loader = torch.load('train_loader_mlpf.pth')

    # train sample model
    model = MLPF(num_convs=2)
    model.train()
    # quick_train(device, model, loader, epochs=4)

    R = []
    # get a sample event for lrp testing
    for i, event in enumerate(loader):
        print(f'Event # {i}')
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

        small_batch = get_small_batch(event, size=size)
        print(f'Testing lrp on: \n {small_batch}')

        # run lrp on sample model
        model.eval()
        lrp_instance = LRP_MLPF(device, model, epsilon=1e-9)
        Rscores0 = lrp_instance.explain(small_batch, neuron_to_explain=0)

        R.append(Rscores0.detach().to('cpu'))
        # print('Checking conservation of Rscores for a random sample')
        # sample = 26
        # print('R_input ', Rscores0[sample].sum().item())
        # print('R_output', model(small_batch)[0][sample][0].item())
        # if i == 2:
        #     break
    with open('Rscores.pkl', 'wb') as f:
        pkl.dump(R, f)
