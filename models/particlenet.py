from torch_geometric.data import Data, DataListLoader, Batch
from torch_geometric.loader import DataLoader

import jetnet
import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

from typing import Optional, Union
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor

try:
    from torch_cluster import knn
except ImportError:
    knn = None

import torch
from torch import nn

from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_cluster import knn_graph

import numpy as np
import torch.nn.functional as F


class ParticleNetEdgeNet(nn.Module):
    def __init__(self, in_size, layer_size):
        super(ParticleNetEdgeNet, self).__init__()

        layers = []

        layers.append(nn.Linear(in_size * 2, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for i in range(2):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.model)


class ParticleNet(nn.Module):
    def __init__(self, node_feat_size, num_classes=5):
        super(ParticleNet, self).__init__()
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes

        self.k = 16
        self.num_edge_convs = 3
        self.kernel_sizes = [64, 128, 256]
        self.fc_size = 256
        self.dropout = 0.1

        self.edge_nets = nn.ModuleList()
        self.edge_convs = nn.ModuleList()

        self.kernel_sizes.insert(0, self.node_feat_size)
        self.output_sizes = np.cumsum(self.kernel_sizes)

        self.edge_nets.append(ParticleNetEdgeNet(self.node_feat_size, self.kernel_sizes[1]))
        self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr="mean"))

        for i in range(1, self.num_edge_convs):
            # adding kernel sizes because of skip connections
            self.edge_nets.append(
                ParticleNetEdgeNet(self.output_sizes[i], self.kernel_sizes[i + 1])
            )
            self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr="mean"))

        self.fc1 = nn.Sequential(nn.Linear(self.output_sizes[-1], self.fc_size))

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

    def forward(self, batch, ret_activations=False, relu_activations=False):
        x = batch.x
        batch = batch.batch

        for i in range(self.num_edge_convs):
            # using only angular coords for knn in first edgeconv block
            edge_index = (
                knn_graph(x[:, :2], self.k, batch) if i == 0 else knn_graph(x, self.k, batch)
            )
            x = torch.cat(
                (self.edge_convs[i](x, edge_index), x), dim=1
            )  # concatenating with original features i.e. skip connection

        x = global_mean_pool(x, batch)
        x = self.fc1(x)

        if ret_activations:
            if relu_activations:
                return F.relu(x)
            else:
                return x  # for Frechet ParticleNet Distance
        else:
            x = self.dropout_layer(F.relu(x))

        return self.fc2(x)  # no softmax because pytorch cross entropy loss includes softmax


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
