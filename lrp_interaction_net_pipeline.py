from utils import H5Data
from models import INTagger
from explainer import LRP_captum

import argparse
from tqdm import tqdm
import matplotlib
from sklearn.metrics import accuracy_score
import pandas as pd
import mplhep as hep
from glob import glob
import sys
import os
import os.path as osp
import pickle as pkl
import matplotlib.pyplot as plt
import itertools
import numpy as np

import torch
import torch.nn as nn


# this script runs lrp on an interaction network

parser = argparse.ArgumentParser()

parser.add_argument("--train",        dest='train',     action='store_true', help="trains a model")
parser.add_argument("--test_acc",     dest='test_acc',  action='store_true', help="tests the accuracy of the model")
parser.add_argument("--batch_size",   type=int,         default=32,          help="number of jets per batch")
parser.add_argument("--stop_at",      type=int,         default=100,         help="run lrp on how many batches of 32 jets")

args = parser.parse_args()


def compute_Rscores(data_val, model, stop_at):
    """
    Given the dataset and model, runs forward passes and computes Rscores using CAPTUM's LRP implementation.

    Args
        data_val: an Hbb-QCD data generator
        model: a pytorch implementation of the interaction network
        stop_at: number of batches to run lrp on
    """

    # initalize an lrp instance based on CAPTUM
    lrp = LRP_captum(model)

    # retrieve a batch just to setup the dimensions of the Rscore tensor placeholders
    for sub_X, sub_Y, sub_Z in data_val.generate_data():
        input_p = torch.tensor(sub_X[2])
        input_sv = torch.tensor(sub_X[3])
        target = torch.tensor(sub_Y[0])
        spec = torch.tensor(sub_Z[0])
        break

    # initialize the Rscore tensors for Higgs & QCD jets
    Rtensor_p_H, Rtensor_p_QCD = torch.zeros(input_p.shape[1], input_p.shape[2]), torch.zeros(input_p.shape[1], input_p.shape[2])
    Rtensor_sv_H, Rtensor_sv_QCD = torch.zeros(input_sv.shape[1], input_sv.shape[2]), torch.zeros(input_sv.shape[1], input_sv.shape[2])
    correct_num_H, correct_num_QCD = 0, 0
    tot_num_H, tot_num_QCD = 0, 0

    # run over the dataset
    for batch, sub in tqdm(enumerate(list(data_val.generate_data()))):
        # unpack the batch because i don't know who set it up this way
        sub_X, sub_Y, sub_Zsub = sub
        input_p = torch.tensor(sub_X[2])
        input_sv = torch.tensor(sub_X[3])
        target = torch.tensor(sub_Y[0])
        spec = torch.tensor(sub_Z[0])

        # run forward pass and lrp per batch
        preds = model(input_p, input_sv)
        Rscores_p_batch, Rscores_sv_batch = lrp.attribute((input_p.requires_grad_(), input_sv.requires_grad_()))

        # unpack the batch
        for jet in range(len(preds)):

            true_class = torch.argmax(target[jet])
            pred_class = preds[jet].round()

            Rscores_p = Rscores_p_batch[jet]
            Rscores_sv = Rscores_sv_batch[jet]

            # absolutize and normalize the Rscores
            Rscores_p = Rscores_p.absolute()
            Rscores_sv = Rscores_sv.absolute()
            if Rscores_p.sum() != 0:
                Rscores_p = Rscores_p / Rscores_p.sum()
            if Rscores_sv.sum() != 0:
                Rscores_sv = Rscores_sv / Rscores_sv.sum()

            # sum the Rscores for Higgs and QCD jets seperately
            if true_class == 1:  # if Higgs
                if (pred_class == true_class):  # check if the node was correctly classified
                    Rtensor_p_H = Rtensor_p_H + Rscores_p
                    Rtensor_sv_H = Rtensor_sv_H + Rscores_sv
                    correct_num_H = correct_num_H + 1
                tot_num_H = tot_num_H + 1
            else:   # if QCD
                if (pred_class == true_class):  # check if the node was correctly classified
                    Rtensor_p_QCD = Rtensor_p_QCD + Rscores_p
                    Rtensor_sv_QCD = Rtensor_sv_QCD + Rscores_sv
                    correct_num_QCD = correct_num_QCD + 1
                tot_num_QCD = tot_num_QCD + 1

        if batch == stop_at:
            break   # bcz my laptop's memory run out :')

    # normalize again
    Rtensor_p_H = Rtensor_p_H / correct_num_H
    Rtensor_sv_H = Rtensor_sv_H / correct_num_H

    Rtensor_p_QCD = Rtensor_p_QCD / correct_num_QCD
    Rtensor_sv_QCD = Rtensor_sv_QCD / correct_num_QCD

    print(f'{correct_num_H} out of {tot_num_H} correctly classified Higgs jets used for lrp')
    print(f'{correct_num_QCD} out of {tot_num_QCD} correctly classified QCD jets used for lrp')

    return Rtensor_p_H, Rtensor_sv_H, Rtensor_p_QCD, Rtensor_sv_QCD


def plot_Rscores(Rtensor_p_H, Rtensor_sv_H, Rtensor_p_QCD, Rtensor_sv_QCD):
    """
    Given the computed Rscores, makes plots of the Rscores for primary, and secondary vertices for Higgs and QCD

    Args
        Rtensor_p_H: the computed Rscore (input_pv-like) tensor of primary vertices for correctly classified Higgs jets
        Rtensor_sv_H: the computed Rscore (input_sv-like) tensor of secondary vertices for correctly classified Higgs jets
        Rtensor_p_QCD: the computed Rscore (input_pv-like) tensor of primary vertices for correctly classified QCD jets
        Rtensor_sv_QCD: the computed Rscore (input_sv-like) tensor of secondary vertices for correctly classified QCD jets
    """

    pos = 2 * np.arange(len(params))
    plt.bar(pos, Rtensor_p_H.sum(axis=1).detach().numpy(), align='center', label='Hbb jet')
    plt.bar(pos + 1, Rtensor_p_QCD.sum(axis=1).detach().numpy(), align='center', label='QCD jet')
    plt.xticks(pos + 0.5, params, rotation='vertical')
    plt.title(f'Rscores for primary vertices of correctly classified Hbb and QCD jets')
    plt.legend()
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig('Rscores_p.pdf')
    plt.show()

    pos = 2 * np.arange(len(params_sv))
    plt.bar(pos, Rtensor_sv_H.sum(axis=1).detach().numpy(), align='center', label='Hbb jet')
    plt.bar(pos + 1, Rtensor_sv_QCD.sum(axis=1).detach().numpy(), align='center', label='QCD jet')
    plt.xticks(pos + 0.5, params_sv, rotation='vertical')
    plt.title(f'Rscores for secondary vertices of correctly classified Hbb and QCD jets')
    plt.legend()
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig('Rscores_sv.pdf')
    plt.show()


if __name__ == "__main__":
    """
    e.g.
    python3 interaction_net_pipeline.py
    """

    # define the primary vertex features
    params = ['track_ptrel',
              'track_erel',
              'track_phirel',
              'track_etarel',
              'track_deltaR',
              'track_drminsv',
              'track_drsubjet1',
              'track_drsubjet2',
              'track_dz',
              'track_dzsig',
              'track_dxy',
              'track_dxysig',
              'track_normchi2',
              'track_quality',
              'track_dptdpt',
              'track_detadeta',
              'track_dphidphi',
              'track_dxydxy',
              'track_dzdz',
              'track_dxydz',
              'track_dphidxy',
              'track_dlambdadz',
              'trackBTag_EtaRel',
              'trackBTag_PtRatio',
              'trackBTag_PParRatio',
              'trackBTag_Sip2dVal',
              'trackBTag_Sip2dSig',
              'trackBTag_Sip3dVal',
              'trackBTag_Sip3dSig',
              'trackBTag_JetDistVal'
              ]

    # define the secondary vertex features
    params_sv = ['sv_ptrel',
                 'sv_erel',
                 'sv_phirel',
                 'sv_etarel',
                 'sv_deltaR',
                 'sv_pt',
                 'sv_mass',
                 'sv_ntracks',
                 'sv_normchi2',
                 'sv_dxy',
                 'sv_dxysig',
                 'sv_d3d',
                 'sv_d3dsig',
                 'sv_costhetasvpv'
                 ]

    # load the dataset
    print('Loading the dataset')
    data_val = H5Data(batch_size=args.batch_size,
                      cache=None,
                      preloading=0,
                      features_name='training_subgroup',
                      labels_name='target_subgroup',
                      spectators_name='spectator_subgroup')
    data_val.set_file_names(['data/hbb/newdata_6.h5'])

    # retrieve a batch just to setup the dimensions and initalize the model
    for sub_X, sub_Y, sub_Z in data_val.generate_data():
        input_p = torch.tensor(sub_X[2])
        input_sv = torch.tensor(sub_X[3])
        target = torch.tensor(sub_Y[0])
        spec = torch.tensor(sub_Z[0])
        break

    # intialize a model
    print('Initializing an interaction network model')
    model = INTagger(pf_dims=input_p.shape[2],
                     sv_dims=input_sv.shape[2],
                     # num_classes=target.shape[1],
                     num_classes=1,
                     pf_features_dims=input_p.shape[1],
                     sv_features_dims=input_sv.shape[1],
                     hidden=5,
                     De=20,
                     Do=24
                     )
    print(model)

    # train an interaction network
    if args.train:
        print('Will run a training...')
        model.train()
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        losses_epoch = []
        losses = []

        # run a training
        for epoch in tqdm(enumerate(range(20))):
            for sub_X, sub_Y, sub_Z in data_val.generate_data():
                input_p = torch.tensor(sub_X[2])
                input_sv = torch.tensor(sub_X[3])
                target = torch.tensor(sub_Y[0])
                spec = torch.tensor(sub_Z[0])
                # forwardprop
                target = torch.argmax(target, axis=1)
                preds = model(input_p, input_sv)
                loss = torch.nn.functional.binary_cross_entropy(preds, target.float())

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.to('cpu').detach().numpy())
            losses_epoch.append(sum(losses) / len(losses))
        fig, ax = plt.subplots()
        ax.plot(range(len(losses_epoch)), losses_epoch, label='train loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Classifier loss')
        ax.legend(loc='best')
        plt.show()
        # # save the model
        # torch.save(model.state_dict(), 'interaction_net_weights.pth')

    # load a pre-trained model if available
    elif os.path.isfile('junk/interaction_net_weights.pth'):
        print('Will load a pre-trained model...')
        state_dict = torch.load('junk/interaction_net_weights.pth')
        model.load_state_dict(state_dict)

    model.eval()

    # test the accuracy of the model
    if args.test_acc:
        print('Will test the accuracy of the model...')
        num = 0
        deno = 0
        for sub_X, sub_Y, sub_Z in data_val.generate_data():
            input_p = torch.tensor(sub_X[2])
            input_sv = torch.tensor(sub_X[3])
            target = torch.tensor(sub_Y[0])
            spec = torch.tensor(sub_Z[0])

            # forwardprop
            preds = model(input_p, input_sv)
            num = num + (preds.round() == torch.argmax(target, axis=1)).sum()
            deno = deno + len(preds)
        acc = num / deno
        print(f'accuracy is {acc}')

    # run lrp to compute Rscores
    print(f'Will run lrp on {args.stop_at} batches of {args.batch_size} jets a batch...')
    Rtensor_p_H, Rtensor_sv_H, Rtensor_p_QCD, Rtensor_sv_QCD = compute_Rscores(data_val, model, stop_at=args.stop_at)

    # plot the Rscores
    print('Will plot Rscores...')
    plot_Rscores(Rtensor_p_H, Rtensor_sv_H, Rtensor_p_QCD, Rtensor_sv_QCD)
