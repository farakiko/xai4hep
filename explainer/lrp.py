import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU


class LRP():

    """
    LRP class that introduces useful helper functions defined on a PyTorch model, and an explain method that runs layerwise-relevance propagation the model.
    Currently supports:
        - Linear and activation layers in the model
        - skip connections provided that they are only input_features skip connections and that they are defined in the following order torch.cat[(input_features, ...)]

    """

    def __init__(self, device, model, epsilon):

        self.device = device
        self.model = model
        self.epsilon = epsilon  # for stability reasons in the lrp-epsilon rule (by default: a very small number)

        # check if the model has any skip connections to accomodate them
        self.skip_connections = self.find_skip_connections()

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

        for name, module in self.model.named_modules():
            # unfold any containers so as to register hooks only for their child modules
            if ('Linear' in str(type(module))) or ('activation' in str(type(module))):
                module.register_forward_hook(get_activation(name))

        # run a forward pass
        self.model.eval()
        preds = self.model(input.to(self.device)).detach()

        # get the activations
        self.activations = activations.to(device).detach()
        self.num_layers = len(activations.keys())
        self.in_features_dim = self.name2layer(list(activations.keys())[0]).in_features

        print(f'Total number of layers (including activation layers): {self.num_layers}')

        # initialize Rscores for skip connections (in case there are any)
        if len(self.skip_connections) != 0:
            self.skip_connections_relevance = 0

        # initialize the Rscores vector using the output predictions
        Rscores = preds[:, neuron_to_explain].reshape(-1, 1)

        # loop over layers in the model to propagate Rscores backward
        for layer_index in range(self.num_layers, 0, -1):
            Rscores = self.explain_single_layer(Rscores, layer_index, neuron_to_explain)

        print("Finished explaining all layers.")

        if len(self.skip_connections) != 0:
            return Rscores + self.skip_connections_relevance

        return Rscores

    def explain_single_layer(self, Rscores_old, layer_index, neuron_to_explain):
        """
        Attempts to explain a single layer in the model by propagating Rscores backwards using the lrp-epsilon rule.

        Args:
            Rscores_old: a vector containing the Rscores, of the current layer, to be propagated backwards
            layer_index: index that corresponds to the position of the layer in the model (see helper functions)
            neuron_to_explain: the index for a particular neuron in the output layer to explain

        Returns:
            Rscores_new: a vector containing the computed Rscores of the previous layer
        """

        # get layer information
        layer_name = self.index2name(layer_index)
        layer = self.name2layer(layer_name)

        # get layer activations
        input = self.activations[layer_name].detach()

        print(f"Explaining layer {self.num_layers+1-layer_index}/{self.num_layers}: {layer}")

        if 'Linear' in str(layer):
            Rscores_new = self.eps_rule(self, layer, input, Rscores_old, neuron_to_explain)
            return Rscores_new
        else:
            print(f"- skipping layer because it's an activation layer")
            print(f"- Rscores do not need to be computed")
            return Rscores_old

    """
    lrp-epsilon rule
    """

    @staticmethod
    def eps_rule(self, layer, x, Rscores_old, neuron_to_explain):
        """
        Implements the lrp-epsilon rule presented in the following reference: https://doi.org/10.1007/978-3-030-28954-6_10.

        Args:
            layer: a torch.nn module with a corresponding weight matrix W
            x: vector containing the activations of the previous layer
            Rscores_old: a vector containing the Rscores, of the current layer, to be propagated backwards
            neuron_to_explain: the index for a particular neuron in the output layer to explain

        Returns:
            Rscores_new: a vector containing the computed Rscores of the previous layer

        The computation is composed of 3 main steps:
            (1) multiply elementwise each column in the matrix W by the vector x to get the matrix Z (we name it Z to be consistent with the reference's notation)
            (2) divide each column in Z by the sum of the its elements
            (3) matrix multiply the Z matrix and Rscores_old vector to obtain the Rscores_new vector
        """

        torch.cuda.empty_cache()

        W = layer.weight.detach()   # get weight matrix
        W = torch.transpose(W, 0, 1)

        # for the output layer, pick the part of the weight matrix connecting only to the neuron you're attempting to explain
        if layer == list(self.model.modules())[-1]:
            W = W[:, neuron_to_explain].reshape(-1, 1)

        # (1) multiply elementwise each column in the matrix W by the vector x to get the Z matrix
        Z = x.unsqueeze(-1) * W     # unsqueeze will add a necessary new dimension to x and then we use broadcasting

        # (2) divide each column in Z by the sum of the its elements
        Z = Z / (Z.sum(axis=1, keepdim=True) + self.epsilon)    # epsilon is introduced for stability (lrp-epsilon rule)

        # (3) matrix multiply Z and Rscores_old to obtain Rscores_new
        Rscores_new = torch.bmm(Z, Rscores_old.unsqueeze(-1)).squeeze()  # we have to use bmm -> batch matrix multiplication

        print('- Finished computing Rscores')

        print('W.device', W.device)
        print('x.device', x.device)
        print('Z.device', Z.device)
        print('R_new.device', Rscores_new.device)
        print('R_old.device', Rscores_old.device)

        # checking conservation of Rscores
        rtol = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for tol in rtol:
            if (torch.allclose(Rscores_old.sum(axis=1), Rscores_new.sum(axis=1), rtol=tol)):
                print(f'- Rscores are conserved up to relative tolerance {str(tol)}')
                break

        if layer in self.skip_connections:
            print('SKIP CONNECTION LAYER')
            # set aside the relevance of the input_features in the skip connection
            # recall: it is assumed that the skip connections are defined in the following order torch.cat[(input_features, ...)] )
            self.skip_connections_relevance = self.skip_connections_relevance + Rscores_new[:, :self.in_features_dim]
            return Rscores_new[:, self.in_features_dim:]
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
            if ('Linear' in str(type(module))):
                explainable_layers.append(module)

        skip_connections = []
        for layer_index in range(len(explainable_layers) - 1):
            if explainable_layers[layer_index].out_features != explainable_layers[layer_index + 1].in_features:
                skip_connections.append(explainable_layers[layer_index + 1])

        return skip_connections
