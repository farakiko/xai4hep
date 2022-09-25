from torch_geometric.data import Data, Dataset
import time
import matplotlib.pyplot as plt
import matplotlib
import torch_geometric
from torch_geometric.loader import DataListLoader, DataLoader
import pandas as pd
import h5py
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from typing import Callable, Optional, Union
from torch_geometric.data import Data, DataListLoader, Batch
from torch_geometric.loader import DataLoader

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

from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_cluster import knn_graph

import numpy as np
import torch.nn.functional as F

try:
    from torch_cluster import knn
except ImportError:
    knn = None


class EdgeConv_lrp(MessagePassing):
    """
    Copied from pytorch_geometric source code, with the following edits
    1. torch.cat([x_i, x_j - x_i], dim=-1)) -> torch.cat([x_i, x_j], dim=-1))
    2. retrieve edge_activations
    """

    def __init__(self, nn: Callable, aggr: str = "max", **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None), self.edge_activations

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:

        # self.edge_activations = self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        # return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        self.edge_activations = self.nn(torch.cat([x_i, x_j], dim=-1))
        return self.nn(torch.cat([x_i, x_j], dim=-1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


class EdgeConvBlock(nn.Module):
    def __init__(self, in_size, layer_size):
        super(EdgeConvBlock, self).__init__()

        layers = []

        layers.append(nn.Linear(in_size * 2, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for i in range(1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.edge_conv = EdgeConv_lrp(nn.Sequential(*layers), aggr="mean")

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)


class ParticleNet(nn.Module):
    def __init__(self, node_feat_size, num_classes=5, k=3):
        super(ParticleNet, self).__init__()
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes

        self.k = k
        self.num_edge_conv_blocks = 3

        self.kernel_sizes = [self.node_feat_size, 64, 128, 256]
        self.input_sizes = np.cumsum(self.kernel_sizes)  # [4, 4+64, 4+64+128, 4+64+128+256]

        self.fc_size = 256

        self.dropout = 0.1
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # define the edgeconvblocks
        self.edge_conv_blocks = nn.ModuleList()
        for i in range(0, self.num_edge_conv_blocks):
            self.edge_conv_blocks.append(EdgeConvBlock(self.input_sizes[i], self.kernel_sizes[i + 1]))
            # self.edge_conv_blocks.append(EdgeConvBlock(self.kernel_sizes[i], self.kernel_sizes[i + 1]))  # if no skip

        # define the fully connected networks (post-edgeconvs)
        self.fc1 = nn.Linear(self.input_sizes[-1], self.fc_size)
        # self.fc1 = nn.Linear(self.kernel_sizes[-1], self.fc_size)  # if no skip
        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

    def forward(self, batch, relu_activations=False):
        x = batch.x
        batch = batch.batch
        edge_activations = {}
        edge_block_activations = {}
        edge_index = {}

        for i in range(self.num_edge_conv_blocks):

            # using only angular coords for knn in first edgeconv block
            edge_index[f"edge_conv_{i}"] = knn_graph(x[:, :2], self.k, batch) if i == 0 else knn_graph(x, self.k, batch)

            out, edge_activations[f"edge_conv_{i}"] = self.edge_conv_blocks[i](x, edge_index[f"edge_conv_{i}"])

            x = torch.cat((out, x), dim=1)  # concatenating with latent features i.e. skip connections per EdgeConvBlock
            # x = out  # if no skip

            edge_block_activations[f"edge_conv_{i}"] = x

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = self.dropout_layer(F.relu(x))
        x = self.fc2(x)

        # no softmax because pytorch cross entropy loss includes softmax
        return x, edge_activations, edge_block_activations, edge_index

#
# # get sample dataset
# dataset = jetnet.datasets.JetNet(jet_type="g")
#
# # load the dataset in a convenient pyg format
# print('Loading training datafiles...')
# loader = DataLoader(torch.load(f"../data/toptagging/test/processed/data_{0}.pt"), batch_size=1, shuffle=True)
#
# for batch in loader:
#     break
#
# model_kwargs = {
#     "node_feat_size": 7,
#     "num_classes": 1,
#     "k": 12,
# }
#
# model = ParticleNet(**model_kwargs)
#
# _, _, _, edge_index = model(batch)
#

# try:
#     state_dict = model.module.state_dict()
# except AttributeError:
#     state_dict = model.state_dict()
# torch.save(state_dict, f'../state_dict.pth')
