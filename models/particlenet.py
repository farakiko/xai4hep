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
        self.num_edge_conv_blocks = 1
        self.kernel_sizes = [self.node_feat_size, 64]
        self.input_sizes = np.cumsum(self.kernel_sizes)     # [4, 4+64, 4+64+128, 4+64+128+256]
        self.fc_size = 256
        self.dropout = 0.1

        # define the edgeconvblocks
        self.edge_conv_blocks = nn.ModuleList()
        for i in range(0, self.num_edge_conv_blocks):
            self.edge_conv_blocks.append(EdgeConvBlock(self.input_sizes[i], self.kernel_sizes[i + 1]))

        # define the fully connected networks (post-edgeconvs)
        # self.fc1 = nn.Sequential(nn.Linear(self.input_sizes[-1], self.fc_size))
        # self.dropout_layer = nn.Dropout(p=self.dropout)
        # self.fc2 = nn.Linear(self.fc_size, self.num_classes)
        self.fc2 = nn.Linear(self.input_sizes[-1], self.num_classes)    # if skip connection
        self.fc2 = nn.Linear(self.kernel_sizes[-1], self.num_classes)    # if no skip connection

        self.kernel_sizes[-1],

    def forward(self, batch, relu_activations=False):
        x = batch.x
        batch = batch.batch
        edge_activations = {}
        edge_block_activations = {}
        edge_block_activations['input'] = {}
        edge_block_activations['output'] = {}
        edge_index = {}

        for i in range(self.num_edge_conv_blocks):
            edge_block_activations['input'][f'{i}'] = x
            # using only angular coords for knn in first edgeconv block
            edge_index['i'] = (
                knn_graph(x[:, :2], self.k, batch) if i == 0 else knn_graph(x, self.k, batch)
            )

            edge_block_activations['output'][f'{i}'], edge_activations[f'edge_conv_{i}'] = self.edge_conv_blocks[i](x, edge_index['i'])

            # x = torch.cat(
            #     (edge_block_activations['output'][f'{i}'], x), dim=1
            # )  # concatenating with latent features i.e. skip connections per EdgeConvBlock

            x = edge_block_activations['output'][f'{i}']

        x = global_mean_pool(x, batch)

        # x = self.fc1(x)
        # x = self.dropout_layer(F.relu(x))
        x = self.fc2(x)
        # return x, msg  # no softmax because pytorch cross entropy loss includes softmax

        return x, edge_activations, edge_block_activations, edge_index['i']  # no softmax because pytorch cross entropy loss includes softmax


model = ParticleNet(node_feat_size=in_features)
model

small_batch = Batch(x=batch.x[:5], y=batch.y[:5], batch=batch.batch[:5], ptr=batch.ptr[:5])
ret, edge_activations, edge_block_activations, edge_index = model(small_batch)


# TODO
num_neighbors = 2
num_nodes = 3

small_batch = Batch(x=batch.x[:num_nodes], y=batch.y[:num_nodes], batch=batch.batch[:num_nodes], ptr=batch.ptr[:num_nodes])
model = ParticleNet(node_feat_size=in_features, k=num_neighbors)
model

R_edges, edge_index = LRP(model, small_batch, activations, num_nodes)


def LRP(model, small_batch, activations, num_nodes):

    # register hooks
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = input[0]
        return hook

    num_convs = 1
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
    for idx in range(model.num_edge_conv_blocks):
        R_tensor_old = redistribute_across_DNN_layer(ret.detach(), model, activations, 'fc2')
        print(f'Rscores after DNN_layer {R_tensor_old.sum()}')
        R_tensor_old = redistribute_across_global_pooling(R_tensor_old, edge_block_activations['output']['0'], num_neighbors)
        print(f'Rscores after global_pooling {R_tensor_old.sum()}')
        R, R_edges[f'{idx}'] = redistribute_edge_conv(edge_index, model, idx, R_tensor_old, edge_block_activations, edge_activations, activations, num_nodes, num_neighbors)
    return R_edges, edge_index

    print(f'Sum of R_scores of the input: {round(R.sum().item(),3)}')
    return edge_index, R_tensor_old1


def redistribute_edge_conv(edge_index, model, idx, R_tensor_old, edge_block_activations, edge_activations, activations, num_nodes, num_neighbors):
    """
    Function that redistributes Rscores over an EdgeConv block.

    takes R_tensor_old ~ (num_nodes, latent_dim_old)
    and returns R_tensor_new ~ (num_nodes, latent_dim_new)
    """
    R_edges = redistribute_across_edge_pooling(R_tensor_old, edge_index, edge_activations[f'edge_conv_{idx}'], num_neighbors)
    print(f'Rscores after edge_pooling {R_edges.sum()}')
    R_scores = redistribute_across_DNN(R_edges, model, idx, activations)
    print(f'Rscores after DNN {R_scores.sum()}')
    R_scores = redistribute_concat_step(edge_index, R_scores, num_nodes, num_neighbors)
    print(f'Rscores after concat step {R_scores.sum()}')
    return R_scores, R_edges


edge_index[1][edge_index[0] == 0]

edge_index[0][edge_index[1] == 0]


edge_index
i = 0
for neighbor in edge_index[0][edge_index[1] == i]:
    print(neighbor)
for neighbor in edge_index[0][edge_index[1] == i]:
    print(neighbor)


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

    print('R_old', R_old.shape)
    # loop over nodes
    for i in range(num_nodes):
        # print(R_new[i])
        # R_new[i] = 0
        # loop over neighbors
        # print('i', i)

        for j in range(k):
            # for neighbor in edge_index[0][edge_index[1] == i]:
            # print('neighbor', neighbor)
            R_new[i] += R_old[(i * k + j), :latent_dim_new]  # selects the neighbors of the nodes
        #     R_new[i] += R_old[neighbor, :latent_dim_new]  # selects the neighbors of the nodes
        #
        # for node in edge_index[0][edge_index[1] != i]:
        #     if node == i:
        #         print('node', node)
        #         R_new[i] += R_old[node, latent_dim_new:]
        # print('-------------------------------')

        # additionally, adds the Rscores of the node itself that has influenced other neighbors
        for x in range(i, num_nodes * k, k):
            if x == i * k + k:  # trick to avoid double counting
                continue
            R_new[i] += R_old[x, latent_dim_new:]
    return R_new


def redistribute_across_DNN(R_tensor_old, model, idx, activations):
    """
    TODO: Function that takes R_tensor_old ~ (num_nodes*k, latent_dim_old)
    and returns R_tensor_new ~ (num_nodes*k, latent_dim_new)

    Follows simple DNN LRP redistribution

    from step 2 to step 3
    """
    R_tensor_new = R_tensor_old
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
            scaledR = R_tensor_new / denominator

            # (3) compute the new Rscores
            R_tensor_new = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * activations[name].detach()

    return R_tensor_new


def redistribute_across_edge_pooling(R_tensor_old, edge_index, edge_activations, k):
    """
    Function that redistributes Rscores from the nodes over the edges.

    takes R_tensor_old ~ (num_nodes, latent_dim)
    and also edge_activations ~ (num_nodes*k, latent_dim)
    and returns R_tensor_new ~ (num_nodes*k, latent_dim)

    Useful to reditsribute the R_scores backward from the averaging of neighboring edges.

    from step 1 to step 2
    """
    epsilon = 1e-9

    num_nodes = R_tensor_old.shape[0]
    latent_dim = R_tensor_old.shape[1]

    R_tensor_new = torch.ones([num_nodes * k, latent_dim])

    # loop over nodes
    for i in range(num_nodes):
        # loop over neighbors

        for j in range(k):
            deno = edge_activations[(i * k):(i * k) + k].sum(axis=0)  # summing the edge_activations node_i (i.e. the edge_activations of the neighboring nodes)

            # redistribute the Rscores of node_i according to how activated each edge_activation was (normalized by deno)
            R_tensor_new[(i * k) + j] = R_tensor_old[i] * edge_activations[(i * k) + j] / (deno + epsilon)

    return R_tensor_new


def redistribute_across_global_pooling(R_tensor_old, x, k):
    """
    Function that redistributes Rscores from the whole jet over the nodes.

    takes R_tensor_old ~ (1, latent_dim)
    and also takes activations x ~ (num_nodes, latent_dim)
    and returns R_tensor_new ~ (num_nodes, latent_dim)

    Useful to reditsribute the R_scores backward from the averaging of all nodes.

    from step 0 to step 1
    """

    num_nodes = x.shape[0]
    latent_dim = R_tensor_old.shape[1]

    R_tensor_new = torch.ones([num_nodes, latent_dim])

    # loop over nodes
    for i in range(num_nodes):
        # loop over neighbors

        deno = x.sum(axis=0)  # summing the activations of all nodes

        # redistribute the Rscores of the whole jet over the nodes (normalized by deno)
        R_tensor_new[i] = R_tensor_old[0] * x[i] / deno
    return R_tensor_new


def redistribute_across_DNN_layer(R_tensor_old, model, activations, layer_name):
    """
    Function that takes R_tensor_old ~ (num_nodes, latent_dim_old)
    and returns R_tensor_new ~ (num_nodes, latent_dim_new)

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
    scaledR = R_tensor_old / denominator

    # (3) compute the new Rscores
    R_tensor_new = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * activations[layer_name]

    return R_tensor_new


def name2layer(model, layer_name):
    """
    Given the name of a layer (e.g. .nn1.3) returns the corresponding torch module (e.g. Linear(...))
    """
    for name, module in model.named_modules():
        if layer_name == name:
            return module
