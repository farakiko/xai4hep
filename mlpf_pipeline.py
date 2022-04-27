from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor
from typing import Optional, Union
import time
from models import MLPF

import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch

# Check if the GPU configuration has been provided
use_gpu = torch.cuda.device_count() > 0

# define the global base device
if use_gpu:
    print(f'Will use {torch.cuda.device_count()} gpu(s)')
    print("GPU model:", torch.cuda.get_device_name(0))
    device = torch.device('cuda:0')
else:
    print('Will use cpu')
    device = torch.device('cpu')

# load the dataset
loader = torch.load('train_loader_mlpf.pth')

for batch in loader:
    print(batch)
    break

# build a smaller batch


def get_small_batch(batch, size):
    small_batch = Batch()
    small_batch.x = batch.x[:size]
    small_batch.ygen = batch.ygen[:size]
    small_batch.ygen_id = batch.ygen_id[:size]
    small_batch.ycand = batch.ycand[:size]
    small_batch.ycand_id = batch.ycand_id[:size]
    small_batch.batch = batch.batch[:size]
    return small_batch


small_batch = get_small_batch(batch, size=1000)
print(small_batch)


def train(model, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses, accuracies = [], []
    losses_val, accuracies_val = [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        losses_batch, accuracies_batch = [], []
        losses_batch_val, accuracies_batch_val = [], []

        model.train()
        for i, batch in enumerate(loader):
            batch = get_small_batch(batch, 100)
            X = batch
            Y_pid = batch.ygen_id
            Y_p4 = batch.ygen

            # Forwardprop
            preds, A, msg_activations = model(X)
            pred_ids_one_hot = preds[:, :6]
            pred_p4 = preds[:, 6:]

            _, pred_ids = torch.max(pred_ids_one_hot, -1)
            _, target_ids = torch.max(Y_pid, -1)

            loss_pid = torch.nn.functional.cross_entropy(pred_ids_one_hot, torch.argmax(Y_pid, axis=1))
            loss_p4 = torch.nn.functional.mse_loss(pred_p4, Y_p4)

            loss = loss_pid + loss_p4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
            losses_batch.append(loss.detach().cpu().item())
        losses.append(np.mean(losses_batch))
    torch.save(model.state_dict(), "weights.pth")
    return losses, losses_val


model = MLPF(num_convs=0)
model.train()


# losses, losses_val = train(model, 3)


class LRP_MLPF():

    """
    Main trick is to realize that substitute the ".lin_s" layers in Gravnet are irrelevant for explanations so shall be skipped
    The hack, however, is to substitute it with the message_passing step
    """

    def __init__(self, device, model, epsilon):

        self.device = device
        self.model = model.to(device)
        self.epsilon = epsilon  # for stability reasons in the lrp-epsilon rule (by default: a very small number)

        # check if the model has any skip connections to accomodate them
        self.skip_connections = self.find_skip_connections()
        self.msg_passing_layers = self.find_msg_passing_layers()

    """
    explanation functions
    """

    def explain(self, input, neuron_to_explain):
        """
        Primary function to call on an LRP instance to start explaining predictions.
        First, it registers hooks and runs a forward pass on the input.
        Then, it attempts to explain the whole model by looping over the layers in the model and invoking the explain_single_layer function.

        Args:
            input: tensor containing the input sample you wish to explain
            neuron_to_explain: the index for a particular neuron in the output layer you wish to explain

        Returns:
            Rscores: a vector containing the relevance scores of the input features
        """

        # register forward hooks to retrieve intermediate activations
        # in simple words, when the forward pass is called, the following dict() will be filled with (key, value) = ("layer_name", activations)
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = input[0]
            return hook

        for name, module in model.named_modules():
            # unfold any containers so as to register hooks only for their child modules (equivalently we are demanding type(module) != nn.Sequential))
            if ('Linear' in str(type(module))) or ('activation' in str(type(module))):
                module.register_forward_hook(get_activation(name))

        # run a forward pass
        model.eval()
        preds, self.A, self.msg_activations = model(input.to(self.device))

        # get the activations
        self.activations = activations
        self.num_layers = len(activations.keys())
        self.in_features_dim = self.name2layer(list(activations.keys())[0]).in_features

        print(f'Total number of layers (including activation layers): {self.num_layers}')

        # initialize Rscores for skip connections (in case there are any)
        if len(self.skip_connections) != 0:
            self.skip_connections_relevance = 0

        # initialize the Rscores tensor using the output predictions
        Rscores = preds[:, neuron_to_explain].reshape(-1, 1).detach()

        # build an Rtensor
        R_tensor = torch.zeros([Rscores.shape[0], Rscores.shape[0], Rscores.shape[1]]).to(self.device)
        for node in range(R_tensor.shape[0]):
            R_tensor[node][node] = Rscores[node].clone()

        # loop over layers in the model to propagate Rscores backward
        for layer_index in range(self.num_layers, 0, -1):
            R_tensor = self.explain_single_layer(R_tensor, layer_index, neuron_to_explain)

        print("Finished explaining all layers.")

        if len(self.skip_connections) != 0:
            return R_tensor + self.skip_connections_relevance

        return R_tensor

    def explain_single_layer(self, R_tensor_old, layer_index, neuron_to_explain):
        """
        Attempts to explain a single layer in the model by propagating Rscores backwards using the lrp-epsilon rule.

        Args:
            R_tensor_old: a vector containing the Rscores, of the current layer, to be propagated backwards
            layer_index: index that corresponds to the position of the layer in the model (see helper functions)
            neuron_to_explain: the index for a particular neuron in the output layer to explain

        Returns:
            Rscores_new: a vector containing the computed Rscores of the previous layer
        """

        # get layer information
        layer_name = self.index2name(layer_index)
        layer = self.name2layer(layer_name)

        # get layer activations
        if layer_name in self.msg_passing_layers.keys():
            print(f"Explaining layer {self.num_layers+1-layer_index}/{self.num_layers}: MessagePassing layer, {layer}")
            input = self.msg_activations[layer_name[:-6]].to(self.device).detach()
            msg_passing_layer = True
        else:
            print(f"Explaining layer {self.num_layers+1-layer_index}/{self.num_layers}: {layer}")
            input = self.activations[layer_name].to(self.device).detach()
            msg_passing_layer = False

        if 'Linear' in str(layer):
            if layer in self.skip_connections:
                R_tensor_new = torch.zeros([R_tensor_old.shape[0], R_tensor_old.shape[0], layer.in_features - self.in_features_dim]).to(self.device)
            elif msg_passing_layer:
                R_tensor_new = torch.zeros([R_tensor_old.shape[0], R_tensor_old.shape[0], R_tensor_old.shape[-1]]).to(self.device)
            else:
                R_tensor_new = torch.zeros([R_tensor_old.shape[0], R_tensor_old.shape[0], layer.in_features]).to(self.device)

            for node in range(R_tensor_old.shape[0]):
                R_tensor_new[node] = self.eps_rule(self, layer, layer_name, input, R_tensor_old[node], neuron_to_explain, msg_passing_layer)

            # checking conservation of Rscores for a given random node (# 17)
            rtol = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            for tol in rtol:
                if (torch.allclose(R_tensor_old[17].sum(), R_tensor_new[17].to('cpu').sum(), rtol=tol)):
                    print(f'- Rscores are conserved up to relative tolerance {str(tol)}')
                    break

            print('- Finished computing Rscores')
            return R_tensor_new

        else:
            print(f"- skipping layer because it's an activation layer")
            print(f"- Rscores do not need to be computed")
            return R_tensor_old

    """
    lrp-epsilon rule
    """

    @staticmethod
    def eps_rule(self, layer, layer_name, x, Rscores_old, neuron_to_explain, msg_passing_layer):
        """
        Implements the lrp-epsilon rule presented in the following reference: https://doi.org/10.1007/978-3-030-28954-6_10.
        The computation is composed of 3 main steps:
            (1) multiply elementwise each column in the matrix W by the vector x to get the matrix Z (we name it Z to be consistent with the reference's notation)
            (2) divide each column in Z by the sum of the its elements
            (3) matrix multiply the Z matrix and Rscores_old vector to obtain the Rscores_new vector

        Can accomodate message_passing layers if the adjacency matrix and the activations before the message_passing are provided.
        The trick (or as we like to call it, the message_passing hack) is in
            (1) transposing the activations to distribute the Rscores over the other dimension (over nodes instead of features)
            (2) use the adjacency matrix as the weight matrix in the standard lrp rule

        Args:
            layer: a torch.nn module with a corresponding weight matrix W
            x: vector containing the activations of the previous layer
            Rscores_old: a vector containing the Rscores, of the current layer, to be propagated backwards
            neuron_to_explain: the index for a particular neuron in the output layer to explain

        Returns:
            Rscores_new: a vector containing the computed Rscores of the previous layer
        """

        torch.cuda.empty_cache()

        if msg_passing_layer:   # message_passing hack
            x = torch.transpose(x, 0, 1)               # transpose the activations to distribute the Rscores over the other dimension (over nodes instead of features)
            W = self.A[layer_name[:-6]].detach()       # use the adjacency matrix as the weight matrix
        else:
            W = layer.weight.detach()  # get weight matrix

        W = torch.transpose(W, 0, 1)    # sanity check of forward pass: (torch.matmul(x, W) + layer.bias) == layer(x)

        # for the output layer, pick the part of the weight matrix connecting only to the neuron you're attempting to explain
        if layer == list(self.model.modules())[-1]:
            W = W[:, neuron_to_explain].reshape(-1, 1)

        # (1) multiply elementwise each column in the matrix W by the vector x to get the Z matrix
        Z = x.unsqueeze(-1) * W     # unsqueeze will add a necessary new dimension to x and then we use broadcasting

        # (2) divide each column in Z by the sum of the its elements
        Z = Z / (Z.sum(axis=1, keepdim=True) + torch.sign((Z.sum(axis=1, keepdim=True))) * self.epsilon)    # epsilon is introduced for stability (lrp-epsilon rule)

        # (3) matrix multiply Z and Rscores_old to obtain Rscores_new
        if msg_passing_layer:  # message_passing hack
            Rscores_old = torch.transpose(Rscores_old, 0, 1)
        Rscores_new = torch.bmm(Z, Rscores_old.unsqueeze(-1)).squeeze()  # we have to use bmm -> batch matrix multiplication

        if layer in self.skip_connections:
            print('SKIP CONNECTION')
            self.skip_connections_relevance = self.skip_connections_relevance + Rscores_new[:, :self.in_features_dim]
            return Rscores_new[:, self.in_features_dim:]

        if msg_passing_layer:  # message_passing hack
            return torch.transpose(Rscores_new, 0, 1)
        return Rscores_new

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

    def find_skip_connections(self):
        """
        Given a torch model, retuns a list of layers with skip connections... the elements are torch modules (e.g. Linear(...))
        """
        explainable_layers = []
        for name, module in self.model.named_modules():
            if 'lin_s' in name:     # for models that are based on Gravnet, skip the lin_s layers
                continue
            if ('Linear' in str(type(module))):
                explainable_layers.append(module)

        skip_connections = []
        for layer_index in range(len(explainable_layers) - 1):
            if explainable_layers[layer_index].out_features != explainable_layers[layer_index + 1].in_features:
                skip_connections.append(explainable_layers[layer_index + 1])

        return skip_connections

    def find_msg_passing_layers(self):
        """
        Returns a list of ".lin_s" layers from model.named_modules() that shall be substituted with message passing
        """
        msg_passing_layers = {}
        for name, module in self.model.named_modules():
            if 'lin_s' in name:     # for models that are based on Gravnet, replace the .lin_s layers with message_passing
                msg_passing_layers[name] = {}

        return msg_passing_layers


lrp_instance = LRP_MLPF(device, model, epsilon=1e-9)
t0 = time.time()
Rscores0 = lrp_instance.explain(small_batch, neuron_to_explain=0)
tf = time.time()
print('time taken to perform lrp: ', tf - t0)

# # ## taking a few samples, we notice that farouk's lrp preserves R-scores more strictly
# # e.g. sample number 7 here
# sample = 45
# # print('R_input_captum   ', R_captum_target0[0].sum().item())
# print('Rfarouk          ', Rscores0[sample].sum().item())
# print('R_output         ', model(small_batch)[0][sample][0].item())


# register forward hooks to retrieve intermediate activations
# in simple words, when the forward pass is called, the following dict() will be filled with (key, value) = ("layer_name", activations)
# activations = {}
#
#
# def get_activation(name):
#     def hook(model, input, output):
#         activations[name] = input[0]
#     return hook
#
#
# for name, module in model.named_modules():
#     # unfold any containers so as to register hooks only for their child modules (equivalently we are demanding type(module) != nn.Sequential))
#     if ('Linear' in str(type(module))) or ('activation' in str(type(module))):
#         module.register_forward_hook(get_activation(name))
#
# pred, A, msg_activations = model(batch)
#
# activations.keys()
#
#
# list(model.named_modules())[2][1].out_features
#
# """
#
# change activations of lin_s layers with corresponding activations
# have a dict that maps the adjacency matrices to the lin_s layers
#
# """
#
#
# def index2name(layer_index):
#     """
#     Given the index of a layer (e.g. 3) returns the name of the layer (e.g. .nn1.3)
#     """
#     layer_name = list(activations.keys())[layer_index - 1]
#     return layer_name
#
#
# def name2layer(layer_name):
#     """
#     Given the name of a layer (e.g. .nn1.3) returns the corresponding torch module (e.g. Linear(...))
#     """
#     for name, module in model.named_modules():
#         if layer_name == (name):
#             return module
# def find_msg_passing_layers():
#     """
#     Given a torch model, retuns a list of layers with skip connections... the elements are torch modules (e.g. Linear(...))
#     """
#     msg_passing_layers = {}
#     for name, module in model.named_modules():
#         if 'lin_s' in name:     # for models that are based on Gravnet, skip the lin_s layers
#             msg_passing_layers[name] = {}
#             msg_passing_layers[name]['A'] = A[name[:-6]]
#             msg_passing_layers[name]['msg_activations'] = msg_activations[name[:-6]]
#     return msg_passing_layers


# CPU
# 1000 ~ 44.917036056518555
# 900 ~ 37.25138521194458
# 800 ~ 29.98154926300049
# 700 ~ 22.84849214553833
# 600 ~ 16.858211278915405
# 500 ~ 12.155266761779785
# 400 ~ 8.115564107894897
# 300 ~ 4.832310914993286
# 200 ~ 2.417971134185791
# 100 ~ 0.7758309841156006
# 50 ~ 0.28067684173583984


# gpu
# 1000 ~ 27.340571880340576
