from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from typing import Callable, Optional, Union
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
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor


try:
    from torch_cluster import knn
except ImportError:
    knn = None

batch_size = 1
out_neuron = 0
in_features = 4

# get sample dataset
in_features = 4
dataset = jetnet.datasets.JetNet(jet_type='g')

# load the dataset in a convenient pyg format
dataset_pyg = []
for data in dataset:
    d = Data(x=data[0], y=data[1])
    dataset_pyg.append(d)

loader = DataLoader(dataset_pyg, batch_size=batch_size, shuffle=False)

for batch in loader:
    break


class EdgeConv_f(MessagePassing):
    """
    Edits:
    1. cat[x_i, x_j - x_i] -> [x_i, x_j]
    2. retrieve edge_activations
    """

    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
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
        return f'{self.__class__.__name__}(nn={self.nn})'


class EdgeConvBlock(nn.Module):
    def __init__(self, in_size, layer_size):
        super(EdgeConvBlock, self).__init__()

        layers = []

        layers.append(nn.Linear(in_size * 2, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for i in range(0):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.edge_conv = EdgeConv_f(nn.Sequential(*layers), aggr="mean")

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)


class ParticleNet(nn.Module):
    def __init__(self, node_feat_size, num_classes=5, k=3):
        super(ParticleNet, self).__init__()
        self.node_feat_size = node_feat_size
        # self.num_classes = num_classes
        self.num_classes = 1  # for now

        self.k = k
        # self.num_edge_conv_blocks = 3
        # self.kernel_sizes = [self.node_feat_size, 64, 128, 256]
        self.num_edge_conv_blocks = 2
        self.kernel_sizes = [self.node_feat_size, 64, 128]
        self.input_sizes = np.cumsum(self.kernel_sizes)     # [4, 4+64, 4+64+128, 4+64+128+256]
        self.fc_size = 256
        self.dropout = 0.1

        # define the edgeconvblocks
        self.edge_conv_blocks = nn.ModuleList()
        for i in range(0, self.num_edge_conv_blocks):
            # self.edge_conv_blocks.append(EdgeConvBlock(self.input_sizes[i], self.kernel_sizes[i + 1]))    # with skip connections
            self.edge_conv_blocks.append(EdgeConvBlock(self.kernel_sizes[i], self.kernel_sizes[i + 1]))

        # define the fully connected networks (post-edgeconvs)
        # self.fc1 = nn.Sequential(nn.Linear(self.input_sizes[-1], self.fc_size)) # if skip connection
        self.fc1 = nn.Linear(self.kernel_sizes[-1], self.fc_size)  # if no skip connection
        # self.dropout_layer = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

    def forward(self, batch, relu_activations=False):
        x = batch.x
        batch = batch.batch
        edge_activations = {}
        edge_block_activations = {}
        edge_index = {}

        for i in range(self.num_edge_conv_blocks):
            # using only angular coords for knn in first edgeconv block
            edge_index[f'edge_conv_{i}'] = (
                knn_graph(x[:, :2], self.k, batch) if i == 0 else knn_graph(x, self.k, batch)
            )

            edge_block_activations[f'edge_conv_{i}'], edge_activations[f'edge_conv_{i}'] = self.edge_conv_blocks[i](x, edge_index[f'edge_conv_{i}'])

            # x = torch.cat(
            #     (edge_block_activations['output'][f'{i}'], x), dim=1
            # )  # concatenating with latent features i.e. skip connections per EdgeConvBlock

            x = edge_block_activations[f'edge_conv_{i}']

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        # x = self.dropout_layer(F.relu(x))
        x = self.fc2(x)

        return x, edge_activations, edge_block_activations, edge_index  # no softmax because pytorch cross entropy loss includes softmax


model = ParticleNet(node_feat_size=in_features)
model

small_batch = Batch(x=batch.x[:5], y=batch.y[:5], batch=batch.batch[:5], ptr=batch.ptr[:5])
ret, edge_activations, edge_block_activations, edge_index = model(small_batch)


# TODO
num_neighbors = 16
num_nodes = 30

small_batch = Batch(x=batch.x[:num_nodes], y=batch.y[:num_nodes], batch=batch.batch[:num_nodes], ptr=batch.ptr[:num_nodes])
model = ParticleNet(node_feat_size=in_features, k=num_neighbors)

R_tensor, R_edges, edge_index = LRP(model, small_batch, num_nodes)


def LRP(model, small_batch, num_nodes):

    # register hooks
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = input[0]
        return hook

    # unpack the EdgeConv Block module to register appropriate hooks for the children modules
    num_convs = model.num_edge_conv_blocks
    for name, module in model.named_modules():
        if 'edge_conv' in name:
            if '.edge_conv' not in name:
                for num in range(num_convs):
                    if f'.{num}' in name:
                        for n, m in module.named_modules():
                            if ('nn' in n) and (n != 'edge_conv.nn'):
                                m.register_forward_hook(get_activation(name + '.' + n))
        elif 'fc' in name:
            module.register_forward_hook(get_activation(name))

    # define parameters
    num_nodes = small_batch.x.shape[0]
    num_neighbors = model.k

    # run forward pass
    ret, edge_activations, edge_block_activations, edge_index = model(small_batch)

    print(f'Sum of R_scores of the output: {round(ret.sum().item(),3)}')

    # run LRP
    R_edges = {}

    Rscores = redistribute_across_fc_layer(ret.detach(), model, activations, 'fc2')
    print(f"Rscores after 'fc2' layer: {Rscores.sum()}")
    Rscores = redistribute_across_fc_layer(Rscores, model, activations, 'fc1')
    print(f"Rscores after 'fc1' layer: {Rscores.sum()}")
    Rscores = redistribute_across_global_pooling(Rscores, edge_block_activations[f'edge_conv_{num_convs-1}'], num_neighbors)
    print(f'Rscores after global_pooling {Rscores.sum()}')

    for idx in range(num_convs - 1, -1, -1):
        Rscores, R_edges[f'edge_conv_{idx}'] = redistribute_edge_conv(edge_index[f'edge_conv_{idx}'], model, idx, Rscores, edge_activations, activations, num_nodes, num_neighbors)

    return Rscores, R_connections, edge_index


def redistribute_edge_conv(edge_index, model, idx, Rscores, edge_activations, activations, num_nodes, num_neighbors):
    """
    Function that redistributes Rscores over an EdgeConv block.

    takes R_tensor_old ~ (num_nodes, latent_dim_old)
    and returns R_tensor_new ~ (num_nodes, latent_dim_new)
    """
    R_edges = redistribute_across_edge_pooling(Rscores, edge_index, edge_activations[f'edge_conv_{idx}'], num_neighbors)
    print(f'Rscores after edge_pooling # {idx}: {R_edges.sum()}')
    R_scores = redistribute_across_DNN(R_edges, model, idx, activations)
    print(f'Rscores after DNN # {idx}: {R_scores.sum()}')
    R_scores = redistribute_concat_step(edge_index, R_scores, num_nodes, num_neighbors)
    print(f'Rscores after concat step # {idx}: {R_scores.sum()}')
    return R_scores, R_edges


def redistribute_concat_step(edge_index, R_old, num_nodes, k):
    """
    Function that takes R_old ~ (num_nodes*k, latent_dim)
    and returns R_new ~ (num_nodes, latent_dim/2)

    Useful to reditsribute the R_scores backward from the concatenation step that happens to perform EdgeConv.
    Note: latent_dim should be an even number as it is a concat of two nodes.

    Assumes that the concat is [x_i, x_j] not [x_i, x_i-x_j]

    from step 3 to step 4
    """
    latent_dim_old = R_old.shape[-1]
    latent_dim_new = int(latent_dim_old / 2)

    R_new = torch.zeros([num_nodes, latent_dim_new])

    # loop over nodes
    for i in range(num_nodes):
        for num_x, x in enumerate(edge_index[1]):
            if i == x:
                R_new[i] += R_old[num_x, :latent_dim_new]

        for num_x, x in enumerate(edge_index[0]):
            if i == x:
                R_new[i] += R_old[num_x, latent_dim_new:]

    return R_new


def redistribute_across_DNN(R_old, model, idx, activations):
    """
    TODO: Function that takes R_old ~ (num_nodes*k, latent_dim_old)
    and returns R_new ~ (num_nodes*k, latent_dim_new)

    Follows simple DNN LRP redistribution

    from step 2 to step 3
    """
    R_new = R_old
    epsilon = 1e-9

    # loop over DNN layers
    for name, layer in model.named_modules():
        if f'edge_conv_blocks.{idx}.edge_conv.nn.' in str(name):
            # print(f'layer: {layer}')
            if 'Linear' not in str(layer):
                continue

            W = layer.weight.detach()  # get weight matrix
            W = torch.transpose(W, 0, 1)    # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

            # (1) compute the denominator
            denominator = torch.matmul(activations[name].detach(), W) + epsilon

            # (2) scale the Rscores
            scaledR = R_new / denominator

            # (3) compute the new Rscores
            R_new = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * activations[name].detach()

    return R_new


def redistribute_across_edge_pooling(R_old, edge_index, edge_activations, k):
    """
    Function that redistributes Rscores from the nodes over the edges.

    takes R_old ~ (num_nodes, latent_dim)
    and also edge_activations ~ (num_nodes*k, latent_dim)
    and returns R_new ~ (num_nodes*k, latent_dim)

    Useful to reditsribute the R_scores backward from the averaging of neighboring edges.

    from step 1 to step 2
    """
    epsilon = 1e-9

    num_nodes = R_old.shape[0]
    latent_dim = R_old.shape[1]

    R_new = torch.ones([num_nodes * k, latent_dim])

    # loop over nodes
    for i in range(num_nodes):
        # loop over neighbors

        for j in range(k):
            deno = edge_activations[(i * k):(i * k) + k].sum(axis=0)  # summing the edge_activations node_i (i.e. the edge_activations of the neighboring nodes)

            # redistribute the Rscores of node_i according to how activated each edge_activation was (normalized by deno)
            R_new[(i * k) + j] = R_old[i] * edge_activations[(i * k) + j] / (deno + epsilon)

    return R_new


def redistribute_across_global_pooling(R_old, x, k):
    """
    Function that redistributes Rscores from the whole jet over the nodes.

    takes R_old ~ (1, latent_dim)
    and also takes activations x ~ (num_nodes, latent_dim)
    and returns R_new ~ (num_nodes, latent_dim)

    Useful to reditsribute the R_scores backward from the averaging of all nodes.

    from step 0 to step 1
    """

    num_nodes = x.shape[0]
    latent_dim = R_old.shape[1]

    R_new = torch.ones([num_nodes, latent_dim])

    # loop over nodes
    for i in range(num_nodes):
        # loop over neighbors

        deno = x.sum(axis=0)  # summing the activations of all nodes

        # redistribute the Rscores of the whole jet over the nodes (normalized by deno)
        R_new[i] = R_old[0] * x[i] / deno
    return R_new


def redistribute_across_fc_layer(R_old, model, activations, layer_name):
    """
    Function that takes R_old ~ (num_nodes, latent_dim_old)
    and returns R_new ~ (num_nodes, latent_dim_new)

    Follows simple DNN LRP redistribution over a given layer.
    """

    epsilon = 1e-9
    # loop over DNN layers
    layer = name2layer(model, layer_name)

    W = layer.weight.detach()  # get weight matrix
    W = torch.transpose(W, 0, 1)    # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

    # (1) compute the denominator
    denominator = torch.matmul(activations[layer_name], W) + epsilon

    # (2) scale the Rscores
    scaledR = R_old / denominator

    # (3) compute the new Rscores
    R_new = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * activations[layer_name]

    return R_new


def name2layer(model, layer_name):
    """
    Given the name of a layer (e.g. .nn1.3) returns the corresponding torch module (e.g. Linear(...))
    """
    for name, module in model.named_modules():
        if layer_name == name:
            return module


def edge_activations_to_dict(edge_index, edge_activations):
    """
    Given the graph structure (defined by edge_index), and the edge activations tensor (retrived from the EdgeConv)
    constructs the dictionary of connections
    """
    R_connections = {}
    for i in range(len(edge_index[0])):
        R_connections[f'{edge_index[1][i]}_{edge_index[0][i]}'] = edge_activations['edge_conv_0'][i]
    return R_connections
