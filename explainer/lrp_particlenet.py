from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
import numpy as np
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch import nn
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor
from typing import Optional, Union
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch.nn import Linear
from torch import Tensor
import jetnet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, DataListLoader, Batch
from typing import Callable, Optional, Union
import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU


class LRP_ParticleNet():

    """
    LRP class that introduces useful helper functions defined on a ParticleNet-based model,
    and an explain method that runs layerwise-relevance propagation the model.
    """

    def __init__(self, device, model, epsilon):

        self.device = device
        self.model = model.to(device)
        self.epsilon = epsilon  # for stability reasons in the lrp-epsilon rule (by default: a very small number)

    """
    explanation functions
    """

    def explain(self, input, neuron_to_explain):
        """
        Primary function to call on an LRP instance to start explaining predictions.
        It registers hooks and runs a forward pass on the input, then it attempts to explain the whole model by looping over EdgeConv blocks.

        Args:
            input: tensor containing the input sample you wish to explain
            neuron_to_explain: the index for a particular neuron in the output layer you wish to explain

        Returns:
            Rscores: a vector containing the relevance scores of the input features
            R_edges: a dictionary containing Rscores for the edges for each EdgeConv block
            edge_index: a dictionary containing the edge_index of each graph constructed during an EdgeConv block
        """

        # register forward hooks to retrieve intermediate activations
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = input[0]
            return hook

        # unpack the EdgeConv Block module to register appropriate hooks for the children modules
        self.num_convs = self.model.num_edge_conv_blocks
        for name, module in self.model.named_modules():
            if 'edge_conv' in name:
                if '.edge_conv' not in name:
                    for num in range(self.num_convs):
                        if f'.{num}' in name:
                            for n, m in module.named_modules():
                                if ('nn' in n) and (n != 'edge_conv.nn'):
                                    m.register_forward_hook(get_activation(name + '.' + n))
            elif 'fc' in name:
                module.register_forward_hook(get_activation(name))

        # define parameters
        self.num_nodes = input.x.shape[0]
        self.num_neighbors = self.model.k

        # run forward pass
        self.model.eval()
        preds, self.edge_activations, self.edge_block_activations, self.edge_index = self.model(input)

        # get the activations
        self.activations = activations

        # initialize the Rscores vector using the output predictions
        Rscores = preds[:, neuron_to_explain].reshape(-1, 1).detach()

        print(f'Sum of R_scores of the output: {round(Rscores.sum().item(),4)}')

        # run LRP
        Rscores = self.redistribute_across_fc_layer(Rscores, 'fc2', neuron_to_explain)
        print(f"Rscores after 'fc2' layer: {round(Rscores.sum().item(),4)}")

        Rscores = self.redistribute_across_fc_layer(Rscores, 'fc1')
        print(f"Rscores after 'fc1' layer: {round(Rscores.sum().item(),4)}")

        Rscores = self.redistribute_across_global_pooling(Rscores)
        print(f'Rscores after global_pooling {round(Rscores.sum().item(),4)}')

        R_edges = {}
        skip_connection_Rscores = None

        # loop over EdgeConv blocks
        for idx in range(self.num_convs - 1, -1, -1):
            Rscores, R_edges[f'edge_conv_{idx}'], skip_connection_Rscores = self.redistribute_EdgeConv(Rscores, idx, skip_connection_Rscores)

        return Rscores, R_edges, self.edge_index

    """
    EdgeConv redistribution
    """

    def redistribute_EdgeConv(self, Rscores, idx, skip_connection_Rscores=None):
        """
        Function that redistributes Rscores over an EdgeConv block.

        takes R_scores ~ (num_nodes, latent_dim_old)
        and returns R_scores ~ (num_nodes, latent_dim_new)
        """
        if skip_connection_Rscores != None:     # if there are skip connections, add the Rscores of those node
            Rscores += skip_connection_Rscores

        # seperate the skip connection nodes from the actual nodes
        skip_connection_Rscores = Rscores[:, self.model.kernel_sizes[idx + 1]:]
        Rscores = Rscores[:, :self.model.kernel_sizes[idx + 1]]

        R_edges = self.redistribute_across_edge_pooling(Rscores, idx)

        print(f'Rscores after edge_pooling # {idx}: {round((R_edges.sum() + skip_connection_Rscores.sum()).item(),4)}')

        R_scores = self.redistribute_across_DNN(R_edges, idx)
        print(f'Rscores after DNN # {idx}: {round((R_scores.sum() + skip_connection_Rscores.sum()).item(),4)}')

        R_scores = self.redistribute_concat_step(R_scores, idx)
        print(f'Rscores after concat step # {idx}: {round((R_scores.sum() + skip_connection_Rscores.sum()).item(),4)}')

        return R_scores, R_edges, skip_connection_Rscores

    """
    special redistribution rules unique to ParticleNet
    """

    def redistribute_concat_step(self, R_old, idx):
        """
        Useful to reditsribute the R_scores backward from the concatenation step that happens to perform EdgeConv.

        Function that takes R_old ~ (num_nodes*k, latent_dim)
        and returns R_new ~ (num_nodes, latent_dim/2)

        Note: latent_dim should be an even number as it is a concat of two node features.
        Note: Assumes that the concat is [x_i, x_j] not [x_i, x_i-x_j]
        """

        latent_dim_old = R_old.shape[-1]
        latent_dim_new = int(latent_dim_old / 2)

        R_new = torch.zeros([self.num_nodes, latent_dim_new])

        # loop over nodes
        for i in range(self.num_nodes):
            for num_x, x in enumerate(self.edge_index[f'edge_conv_{idx}'][1]):
                if i == x:
                    R_new[i] += R_old[num_x, :latent_dim_new]

            for num_x, x in enumerate(self.edge_index[f'edge_conv_{idx}'][0]):
                if i == x:
                    R_new[i] += R_old[num_x, latent_dim_new:]

        return R_new

    def redistribute_across_edge_pooling(self, R_old, idx):
        """
        Useful to reditsribute the R_scores backward from the averaging of neighboring edges. It redistributes Rscores from the nodes over the edges.

        takes R_old ~ (num_nodes, latent_dim) and edge_activations ~ (num_nodes*k, latent_dim)
        returns R_new ~ (num_nodes*k, latent_dim)
        """

        latent_dim = R_old.shape[1]

        R_new = torch.ones([self.num_nodes * self.num_neighbors, latent_dim])

        # loop over nodes
        for i in range(self.num_nodes):
            # loop over neighbors

            for j in range(self.num_neighbors):
                deno = self.edge_activations[f'edge_conv_{idx}'][(i * self.num_neighbors):(i * self.num_neighbors) + self.num_neighbors].sum(axis=0)  # summing the edge_activations node_i (i.e. the edge_activations of the neighboring nodes)

                # redistribute the Rscores of node_i according to how activated each edge_activation was (normalized by deno)
                R_new[(i * self.num_neighbors) + j] = R_old[i] * self.edge_activations[f'edge_conv_{idx}'][(i * self.num_neighbors) + j] / (deno + self.epsilon)

        return R_new

    def redistribute_across_global_pooling(self, R_old):
        """
        Useful to reditsribute the R_scores backward from the averaging of all nodes. It redistributes Rscores from the whole jet over the nodes.

        takes R_old ~ (1, latent_dim)
        and returns R_new ~ (num_nodes, latent_dim)
        """

        latent_dim = R_old.shape[1]

        R_new = torch.ones([self.num_nodes, latent_dim])
        x = self.edge_block_activations[f'edge_conv_{self.num_convs-1}']

        # loop over nodes
        for i in range(self.num_nodes):

            deno = x.sum(axis=0) + self.epsilon   # summing the activations of all nodes

            # redistribute the Rscores of the whole jet over the nodes (normalized by deno)
            R_new[i] = R_old[0] * x[i] / deno
        return R_new

    """
    lrp-epsilon rule
    """

    def redistribute_across_DNN(self, R_old, idx):
        """
        Implements the lrp-epsilon rule presented in the following reference: https://doi.org/10.1007/978-3-030-28954-6_10.
        Follows simple DNN LRP redistribution over a the Sequential FC layers of a given EdgeConv DNN.

        takes R_old ~ (num_nodes*k, latent_dim_old)
        and returns R_new ~ (num_nodes*k, latent_dim_new)
        """

        layer_names = []

        # loop over DNN layers
        for name, layer in self.model.named_modules():
            if f'edge_conv_blocks.{idx}.edge_conv.nn.' in str(name):
                if 'Linear' not in str(layer):
                    continue
                layer_names.append(name)

        layer_names.reverse()

        R_new = R_old
        for name in layer_names:
            layer = self.name2layer(name)

            W = layer.weight.detach()  # get weight matrix
            W = torch.transpose(W, 0, 1)    # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

            # (1) compute the denominator
            denominator = torch.matmul(self.activations[name].detach(), W) + self.epsilon

            # (2) scale the Rscores
            scaledR = R_new / denominator

            # (3) compute the new Rscores
            R_new = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * self.activations[name].detach()

        return R_new

    def redistribute_across_fc_layer(self, R_old, layer_name, neuron_to_explain=None):
        """
        Implements the lrp-epsilon rule presented in the following reference: https://doi.org/10.1007/978-3-030-28954-6_10.
        Follows simple DNN LRP redistribution over a single FC layer.

        takes R_old ~ (num_nodes, latent_dim_old)
        and returns R_new ~ (num_nodes, latent_dim_new)
        """

        # loop over DNN layers
        layer = self.name2layer(layer_name)

        W = layer.weight.detach()  # get weight matrix
        W = torch.transpose(W, 0, 1)    # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

        # for the output layer, pick the part of the weight matrix connecting only to the neuron you're attempting to explain
        if neuron_to_explain != None:
            W = W[:, neuron_to_explain].reshape(-1, 1)

        # (1) compute the denominator
        denominator = torch.matmul(self.activations[layer_name], W) + self.epsilon

        # (2) scale the Rscores
        scaledR = R_old / denominator

        # (3) compute the new Rscores
        R_new = torch.matmul(scaledR, torch.transpose(W, 0, 1)) * self.activations[layer_name]

        return R_new

    """
    helper functions
    """

    def index2name(self, layer_index):
        """
        Given the index of a layer (e.g. 3) returns the name of the layer (e.g. .nn1.3)
        """
        layer_name = list(self.activations.keys())[layer_index - 1]
        return layer_name

    def name2layer(self, layer_name):
        """
        Given the name of a layer (e.g. .nn1.3) returns the corresponding torch module (e.g. Linear(...))
        """
        for name, module in self.model.named_modules():
            if layer_name == name:
                return module
