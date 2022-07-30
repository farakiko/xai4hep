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

from ..inits import reset

try:
    from torch_cluster import knn
except ImportError:
    knn = None

batch_size = 1
out_neuron = 0
in_features = 4


class EdgeConv_f(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        self.msg = self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class EdgeConvBlock(nn.Module):
    def __init__(self, in_size, layer_size):
        super(EdgeConvBlock, self).__init__()

        layers = []

        layers.append(nn.Linear(in_size * 2, layer_size))
        # layers.append(nn.BatchNorm1d(layer_size))
        # layers.append(nn.ReLU())

        for i in range(0):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.edge_conv = EdgeConv_f(nn.Sequential(*layers), aggr="mean")

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)


class ParticleNet(nn.Module):
    def __init__(self, node_feat_size, num_classes=5):
        super(ParticleNet, self).__init__()
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes

        self.k = 3
        # self.num_edge_conv_blocks = 3
        # self.kernel_sizes = [self.node_feat_size, 64, 128, 256]
        self.num_edge_conv_blocks = 2
        self.kernel_sizes = [self.node_feat_size, 64, 13]
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
        self.fc2 = nn.Linear(self.input_sizes[-1], self.num_classes)

        self.kernel_sizes[-1],

    def forward(self, batch, relu_activations=False):
        x = batch.x
        batch = batch.batch

        for i in range(self.num_edge_conv_blocks):
            # using only angular coords for knn in first edgeconv block
            edge_index = (
                knn_graph(x[:, :2], self.k, batch) if i == 0 else knn_graph(x, self.k, batch)
            )
            x = torch.cat(
                (self.edge_conv_blocks[i](x, edge_index), x), dim=1
            )  # concatenating with latent features i.e. skip connections per EdgeConvBlock

        x = global_mean_pool(x, batch)

        # x = self.fc1(x)
        # x = self.dropout_layer(F.relu(x))
        x = self.fc2(x)
        # return x, msg  # no softmax because pytorch cross entropy loss includes softmax

        return x  # no softmax because pytorch cross entropy loss includes softmax


model = ParticleNet(node_feat_size=in_features)
model

small_batch = Batch(x=batch.x[:5], y=batch.y[:5], batch=batch.batch[:5], ptr=batch.ptr[:5])
model(small_batch)


for i, name in enumerate(model.modules()):
    if i == 4:
        print(name)
        break


for name, module in model.named_modules():
    # unfold any containers so as to register hooks only for their child modules (equivalently we are demanding type(module) != nn.Sequential))
    if ('Linear' in str(type(module))) or ('activation' in str(type(module))) or ('BatchNorm1d' in str(type(module))):
        print('no')
    else:
        print(name, module)


for name in model.children():
    if 'ModuleList' in str(type(name)):
        print(model.children())


name


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = input[0]
    return hook


num_convs = 2
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


for key, value in activations.items():
    for name, module in model.named_modules():
        if key == name:
            a = module
            break
    print(key, '---', value.shape, '---', a)

model(small_batch)


for key, value in activations.items():
    print(key, value.shape)


activations.keys()
activations['edge_conv_blocks.0.edge_conv.nn.0'].shape
activations['edge_conv_blocks.0.edge_conv.nn.1'].shape
activations['edge_conv_blocks.0.edge_conv.nn.2'].shape
activations['edge_conv_blocks.0.edge_conv.nn'].shape
activations['edge_conv_blocks.0.edge_conv'].shape
activations['edge_conv_blocks.0'].shape
activations['fc2'].shape

activations['edge_conv_blocks.0.edge_conv.nn.2'][:2].sum(axis=0) / 2


activations['fc2']
activations['edge_conv_blocks.0.edge_conv.nn.2'].sum(axis=0) / len(activations['edge_conv_blocks.0.edge_conv.nn.2'])


activations['edge_conv_blocks.0.edge_conv.nn']
activations['edge_conv_blocks.0.edge_conv.nn.0']

activations['edge_conv_blocks.0.edge_conv']
activations['edge_conv_blocks.0']

small_batch.x

# module.register_forward_hook(get_activation(name))

#
# if __name__ == "__main__":
#     # Check if the GPU configuration and define the global base device
#     if torch.cuda.device_count() > 0:
#         print(f'Will use {torch.cuda.device_count()} gpu(s)')
#         print("GPU model:", torch.cuda.get_device_name(0))
#         device = torch.device('cuda:0')
#     else:
#         print('Will use cpu')
#         device = torch.device('cpu')
#
#     # get sample dataset
#     in_features = 4
#     dataset = jetnet.datasets.JetNet(jet_type='g')
#
#     # load the dataset in a convenient pyg format
#     dataset_pyg = []
#     for data in dataset:
#         d = Data(x=data[0], y=data[1])
#         dataset_pyg.append(d)
#
#     loader = DataLoader(dataset_pyg, batch_size=batch_size, shuffle=False)
#
#     for batch in loader:
#         break
#
#     # # train sample model
#     model = ParticleNet(node_feat_size=in_features)
#     model.train()
#     quick_train(device, model, loader, epochs=4)
#
#     Rtensors_list, preds_list, inputs_list = [], [], []
#
#     # get a sample event for lrp testing
#     for i, event in enumerate(loader):
#         print(f'Explaining event # {i}')
#         # break it down to a smaller part for lrp (to avoid memory issues)
#
#         def get_small_batch(event, size):
#             small_batch = Batch()
#             small_batch.x = event.x[:size]
#             small_batch.ygen = event.ygen[:size]
#             small_batch.ygen_id = event.ygen_id[:size]
#             small_batch.ycand = event.ycand[:size]
#             small_batch.ycand_id = event.ycand_id[:size]
#             small_batch.batch = event.batch[:size]
#             return small_batch
#
#         event = get_small_batch(event, size=size)
#
#         # run lrp on sample model
#         model.eval()
#         lrp_instance = LRP_MLPF(device, model, epsilon=1e-9)
#         Rtensor, pred, input = lrp_instance.explain(event, neuron_to_explain=out_neuron)
#
#         Rtensors_list.append(Rtensor.detach().to('cpu'))
#         preds_list.append(pred.detach().to('cpu'))
#         inputs_list.append(input.detach().to('cpu').to_dict())
#
#         # print('Checking conservation of Rscores for a random sample')
#         # sample = 26
#         # print('R_input ', Rtensor[sample].sum().item())
#         # print('R_output', model(small_batch)[0][sample][0].item())
#         if i == 2:
#             break
#     with open('Rtensors_list.pkl', 'wb') as f:
#         pkl.dump(Rtensors_list, f)
#     with open('preds_list.pkl', 'wb') as f:
#         pkl.dump(preds_list, f)
#     with open('inputs_list.pkl', 'wb') as f:
#         pkl.dump(inputs_list, f)
#
#
# R_old.shape
#
# R_old = torch.ones([num_nodes * k, 6])

# It is assumed that EdgeConv concats (x_i, x_j) not (x_i, x_i - x_j)


def redistribute_concat_step(R_old, num_nodes, k):
    """
    Function that takes R_old ~ (num_nodes*k, latent_dim)
    and returns R_new ~ (num_nodes, latent_dim/2)

    Useful to reditsribute the R_scores backward from the concatenation step that happens to perform EdgeConv.
    Note: latent_dim should be an even number as it is a concat of two nodes.

    from step 3 to step 4
    """
    latent_dim_old = R_old.shape[-1]
    latent_dim_new = int(latent_dim_old / 2)

    R_new = torch.ones([num_nodes, latent_dim_new])
    # loop over nodes
    for i in range(num_nodes):
        R_new[i] = 0
        # loop over neighbors
        for j in range(k):
            R_new[i] += R_old[(i * k + j), :latent_dim_new]  # selects the neighbors of the nodes

        # additionally, adds the Rscores of the node itself that has influenced other neighbors
        for x in range(i, num_nodes * k, k):
            if x == i * k + k:  # trick to avoid double counting
                continue
            R_new[i] += R_old[x, latent_dim_new:]
    return R_new


def redistribute_across_DNN(R_tensor_old, DNN_layers):
    """
    TODO: Function that takes R_tensor_old ~ (num_nodes*k, latent_dim_old)
    and returns R_tensor_new ~ (num_nodes*k, latent_dim_new)

    Follows simple DNN LRP redistribution

    from step 2 to step 3
    """
    R_tensor_new = R_tensor_old
    # loop over DNN layers
    for layer in DNN_layers:
        W = layer.weight.detach()  # get weight matrix
        W = torch.transpose(W, 0, 1)    # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

        # (1) compute the denominator
        denominator = torch.matmul(x[layer], W) + self.epsilon

        # (2) scale the Rscores
        scaledR = R_tensor_new / denominator

        # (3) compute the new Rscores
        R_tensor_new = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * x[layer]

    return R_tensor_new


def redistribute_across_edge_pooling(R_tensor_old, edge_activations, k):
    """
    Function that redistributes Rscores from the nodes over the edges.

    takes R_tensor_old ~ (num_nodes, latent_dim)
    and also edge_activations ~ (num_nodes*k, latent_dim)
    and returns R_tensor_new ~ (num_nodes*k, latent_dim)

    Useful to reditsribute the R_scores backward from the averaging of neighboring edges.

    from step 1 to step 2
    """

    num_nodes = R_tensor_old.shape[0]
    latent_dim = R_tensor_old.shape[1]

    R_tensor_new = torch.ones([num_nodes * k, latent_dim])

    # loop over nodes
    for i in range(num_nodes):
        # loop over neighbors
        for j in range(k):
            deno = edge_activations[(i * k):(i * k) + k].sum(axis=0)  # summing the edge_activations node_i (i.e. the edge_activations of the neighboring nodes)

            # redistribute the Rscores of node_i according to how activated each edge_activation was (normalized by deno)
            R_tensor_new[(i * k) + j] = R_tensor_old[i] * edge_activations[(i * k) + j] / deno
    return R_tensor_new


def redistribute_edge_conv(R_tensor_old, edge_activations, DNN_layers, num_nodes, k):
    """
    Function that redistributes Rscores over an EdgeConv block.

    takes R_tensor_old ~ (num_nodes, latent_dim_old)
    and returns R_tensor_new ~ (num_nodes, latent_dim_new)
    """
    R_tensor_new = redistribute_across_edge_pooling(R_tensor_old, edge_activations, k)
    R_tensor_new = redistribute_across_DNN(R_tensor_new, DNN_layers)
    R_tensor_new = redistribute_concat_step(R_tensor_new, num_nodes, k)
    return R_tensor_new


def redistribute_across_global_pooling(R_tensor_old, x, k):
    """
    Function that redistributes Rscores from the whole jeto over the nodes.

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
