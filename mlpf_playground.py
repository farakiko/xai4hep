import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob
import time

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
import argparse

from explainer import LRP_MLPF, LRP_MLPF_old
from models import MLPF

# this script builds a toy dataset, trains a simple FFN model on the dataset, and tests LRP

"""
python3 -u mlpf_playground.py --size=400
"""
parser = argparse.ArgumentParser()
parser.add_argument("--size",     type=int,           default=100,      help="size of event")
args = parser.parse_args()

size = args.size
out_neuron = 0


def quick_train(device, model, loader, epochs):
    print('Training a model')

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses, accuracies = [], []
    losses_val, accuracies_val = [], []

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        losses_batch, accuracies_batch = [], []
        losses_batch_val, accuracies_batch_val = [], []

        model.train()
        for i, batch in enumerate(loader):
            X = batch
            Y_pid = batch.ygen_id
            Y_p4 = batch.ygen

            # Forwardprop
            preds, A, msg_activations = model(X.to(device))
            pred_ids_one_hot = preds[:, :6]
            pred_p4 = preds[:, 6:]

            _, pred_ids = torch.max(pred_ids_one_hot, -1)
            _, target_ids = torch.max(Y_pid, -1)

            loss_pid = torch.nn.functional.cross_entropy(pred_ids_one_hot, torch.argmax(Y_pid, axis=1).to(device))
            loss_p4 = torch.nn.functional.mse_loss(pred_p4, Y_p4.to(device))

            loss = loss_pid + loss_p4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_batch.append(loss.detach().cpu().item())
        losses.append(np.mean(losses_batch))
    torch.save(model.state_dict(), "weights.pth")
    return losses, losses_val


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
    loader = torch.load('test_loader.pth')

    # train sample model
    model = MLPF(num_convs=2)
    # model.train()
    # quick_train(device, model, loader, epochs=4)

    Rtensors_list, preds_list, inputs_list = [], [], []

    # get a sample event for lrp testing
    for i, event in enumerate(loader):
        print(f'Explaining event # {i}')

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

        event = get_small_batch(event, size=size)
        print(f'Testing lrp on: \n {event}')

        # run lrp on sample model
        model.eval()
        lrp_instance = LRP_MLPF_old(device, model, epsilon=1e-9)
        t0 = time.time()
        Rtensor_old, pred, input = lrp_instance.explain(event, neuron_to_explain=out_neuron)
        t1 = time.time()
        t_old = t1 - t0

        lrp_instance = LRP_MLPF(device, model, epsilon=1e-9)
        t0 = time.time()
        Rtensor_new, pred, input = lrp_instance.explain(event, neuron_to_explain=out_neuron)
        t1 = time.time()
        t_new = t1 - t0

        print(f'-----------------------------------------------')
        print(f'Tested lrp on with size {size}')
        print(f'Old time {t_old}')
        print(f'New time {t_new}')

        # Rtensors_list.append(Rtensor.detach().to('cpu'))
        # preds_list.append(pred.detach().to('cpu'))
        # inputs_list.append(input.detach().to('cpu').to_dict())

        # print('Checking conservation of Rscores for a random sample')
        # sample = 26
        # print('R_input ', Rtensor[sample].sum().item())
        # print('R_output', model(small_batch)[0][sample][0].item())

        # with open('Rtensors_list_old.pkl', 'wb') as f:
        #     pkl.dump(Rtensors_list, f)
        # with open('preds_list.pkl', 'wb') as f:
        #     pkl.dump(preds_list, f)
        # with open('inputs_list.pkl', 'wb') as f:
        #     pkl.dump(inputs_list, f)
        break
