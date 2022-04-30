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
from torch_geometric.data import Data, DataListLoader, Batch
from torch_geometric.loader import DataLoader
import jetnet

# from explainer import LRP_MLPF
from models import ParticleNet

batch_size = 100
out_neuron = 0

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
    in_features = 4
    dataset = jetnet.datasets.JetNet(jet_type='g')

    # load the dataset in a convenient pyg format
    dataset_pyg = []
    for data in dataset:
        d = Data(x=data[0], y=data[1])
        dataset_pyg.append(d)

    loader = DataLoader(dataset_pyg, batch_size=batch_size, shuffle=False)

    # # train sample model
    model = ParticleNet(node_feat_size=in_features)
    # model.train()
    # quick_train(device, model, loader, epochs=4)

    # Rtensors_list, preds_list, inputs_list = [], [], []
    #
    # # get a sample event for lrp testing
    # for i, event in enumerate(loader):
    #     print(f'Explaining event # {i}')
    #     # break it down to a smaller part for lrp (to avoid memory issues)
    #
    #     def get_small_batch(event, size):
    #         small_batch = Batch()
    #         small_batch.x = event.x[:size]
    #         small_batch.ygen = event.ygen[:size]
    #         small_batch.ygen_id = event.ygen_id[:size]
    #         small_batch.ycand = event.ycand[:size]
    #         small_batch.ycand_id = event.ycand_id[:size]
    #         small_batch.batch = event.batch[:size]
    #         return small_batch
    #
    #     event = get_small_batch(event, size=size)
    #
    #     # run lrp on sample model
    #     model.eval()
    #     lrp_instance = LRP_MLPF(device, model, epsilon=1e-9)
    #     Rtensor, pred, input = lrp_instance.explain(event, neuron_to_explain=out_neuron)
    #
    #     Rtensors_list.append(Rtensor.detach().to('cpu'))
    #     preds_list.append(pred.detach().to('cpu'))
    #     inputs_list.append(input.detach().to('cpu').to_dict())
    #
    #     # print('Checking conservation of Rscores for a random sample')
    #     # sample = 26
    #     # print('R_input ', Rtensor[sample].sum().item())
    #     # print('R_output', model(small_batch)[0][sample][0].item())
    #     if i == 2:
    #         break
    # with open('Rtensors_list.pkl', 'wb') as f:
    #     pkl.dump(Rtensors_list, f)
    # with open('preds_list.pkl', 'wb') as f:
    #     pkl.dump(preds_list, f)
    # with open('inputs_list.pkl', 'wb') as f:
    #     pkl.dump(inputs_list, f)
model
