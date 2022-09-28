import vector
import fastjet
import awkward as ak
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from tqdm.notebook import tqdm
from itertools import cycle
import matplotlib.colors as colors
from networkx import draw_networkx_nodes, draw_networkx_edges
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataListLoader, DataLoader
import torch_geometric
import torch.nn as nn
import torch
from models import ParticleNet
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep

import pickle as pkl
import sys
sys.path.insert(0, '..')


# this script runs lrp on a trained ParticleNet model

parser = argparse.ArgumentParser()

parser.add_argument("--N", type=int, default=15, help="Top N edges to look at")

args = parser.parse_args()


def get_subjets(px, py, pz, e):
    vector.register_awkward()
    px = ak.from_regular(ak.from_numpy(px.numpy()))
    py = ak.from_regular(ak.from_numpy(py.numpy()))
    pz = ak.from_regular(ak.from_numpy(pz.numpy()))
    e = ak.from_regular(ak.from_numpy(e.numpy()))

    # define jet directly not an array of jets
    jet = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": e,
            "particle_idx": ak.local_index(px),
            "subjet_idx": ak.zeros_like(px, dtype=int) - 1,
        },
        with_name="Momentum4D",
    )

    pseudojets = []
    pseudojets.append(
        [
            fastjet.PseudoJet(particle.px, particle.py, particle.pz, particle.E)
            for particle in jet
        ]
    )

    subjet_indices = []
    mapping = [jet.subjet_idx.to_list()]   # added square brackets
    for ijet, pseudojet in enumerate(pseudojets):
        subjet_indices.append([])
        cluster = fastjet.ClusterSequence(pseudojet, jetdef)

        # cluster jets
        jets = cluster.inclusive_jets()
        assert len(jets) == 1

        # get the 3 exclusive jets
        subjets = cluster.exclusive_subjets(jets[0], N_SUBJETS)
        assert len(subjets) == N_SUBJETS

        # sort by pt
        subjets = sorted(subjets, key=lambda x: x.pt(), reverse=True)

        for subjet_idx, subjet in enumerate(subjets):
            subjet_indices[-1].append([])
            for subjet_const in subjet.constituents():
                for idx, jet_const in enumerate(pseudojet):
                    if (
                        subjet_const.px() == jet_const.px()
                        and subjet_const.py() == jet_const.py()
                        and subjet_const.pz() == jet_const.pz()
                        and subjet_const.E() == jet_const.E()
                    ):
                        subjet_indices[-1][-1].append(idx)

        for subjet_idx, subjet in enumerate(subjets):
            local_mapping = np.array(mapping[ijet])
            local_mapping[subjet_indices[ijet][subjet_idx]] = subjet_idx
            mapping[ijet] = local_mapping

    # update array
    # array.subjet_idx = ak.Array(mapping)
    jet = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": e,
            "particle_idx": ak.local_index(px),
            "subjet_idx": ak.Array(mapping[0]),  # pick first element
        },
        with_name="Momentum4D",
    )

    jet_vector = vector.obj(
        px=ak.sum(jet.px, axis=-1),
        py=ak.sum(jet.py, axis=-1),
        pz=ak.sum(jet.pz, axis=-1),
        E=ak.sum(jet.E, axis=-1),
    )
    subjet_vectors = [
        vector.obj(
            px=ak.sum(jet.px[jet.subjet_idx == j], axis=-1),
            py=ak.sum(jet.py[jet.subjet_idx == j], axis=-1),
            pz=ak.sum(jet.pz[jet.subjet_idx == j], axis=-1),
            E=ak.sum(jet.E[jet.subjet_idx == j], axis=-1),
        )
        for j in range(0, N_SUBJETS)
    ]

    deta = jet.deltaeta(jet_vector)
    dphi = jet.deltaphi(jet_vector)
    dpt = jet.pt / jet_vector.pt

    subjet_idx = jet.subjet_idx.to_numpy()

    return subjet_idx, subjet_vectors, deta, dphi, dpt


if __name__ == "__main__":
    """
    e.g. to run on prp
    python -u scaling_up.py --N=15

    """

    if torch.cuda.device_count():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    N_SUBJETS = 3

    LABEL = {}
    LABEL[fastjet.kt_algorithm] = "$k_{{\mathrm{{T}}}}$"
    LABEL[fastjet.antikt_algorithm] = "anti-$k_{{\mathrm{{T}}}}$"
    LABEL[fastjet.cambridge_algorithm] = "CA"

    # JET_ALGO = fastjet.kt_algorithm
    # JET_ALGO = fastjet.antikt_algorithm
    JET_ALGO = fastjet.cambridge_algorithm

    jetdef = fastjet.JetDefinition(JET_ALGO, 0.8)

    Top_N = args.N

    top_same = np.array([0] * Top_N)
    top_diff = np.array([0] * Top_N)
    qcd_same = np.array([0] * Top_N)
    qcd_diff = np.array([0] * Top_N)

    PATH = "/xai4hepvol/ParticleNet_6/Rscores_best_10k"

    print('Loading Rscores...')
    with open(f'{PATH}/batch_x.pkl', 'rb') as handle:
        batch_x_list = pkl.load(handle)
    with open(f'{PATH}/batch_y.pkl', 'rb') as handle:
        batch_y_list = pkl.load(handle)
    with open(f'{PATH}/R_edges.pkl', 'rb') as handle:
        R_edges_list = pkl.load(handle)
    with open(f'{PATH}/edge_index.pkl', 'rb') as handle:
        edge_index_list = pkl.load(handle)

    with open(f'{PATH}/batch_px.pkl', 'rb') as handle:
        batch_px_list = pkl.load(handle)
    with open(f'{PATH}/batch_py.pkl', 'rb') as handle:
        batch_py_list = pkl.load(handle)
    with open(f'{PATH}/batch_pz.pkl', 'rb') as handle:
        batch_pz_list = pkl.load(handle)
    with open(f'{PATH}/batch_E.pkl', 'rb') as handle:
        batch_E_list = pkl.load(handle)

    Num_jets = len(batch_px_list)
    for i in range(Num_jets):
        print(f"processing jet # {i}/{Num_jets}")
        jet_label = batch_y_list[i]

        R_edges = R_edges_list[i]['edge_conv_2']
        edge_index_dic = edge_index_list[i]['edge_conv_2']

        px = batch_px_list[i]
        py = batch_py_list[i]
        pz = batch_pz_list[i]
        e = batch_E_list[i]

        # get subjets
        try:
            subjet_idx, subjet_vectors, eta, phi, pt = get_subjets(px, py, pz, e)
        except:
            print(f"skipping jet # {i}")
            continue

        # top N edges
        edge_index, edge_weight = edge_index_dic, torch.abs(R_edges).sum(axis=1)
        edge_Rscores = edge_weight / sum(edge_weight)  # normalize sum of Rscores of all jets to be 1

        for N in range(Top_N):
            for edge in torch.topk(edge_Rscores, N + 1).indices:   # N=0 doesn't make sense here
                if jet_label == 1:
                    if subjet_idx[edge_index[0][edge]] != subjet_idx[edge_index[1][edge]]:
                        top_diff[N] += 1
                    else:
                        top_same[N] += 1
                else:
                    if subjet_idx[edge_index[0][edge]] != subjet_idx[edge_index[1][edge]]:
                        qcd_diff[N] += 1
                    else:
                        qcd_same[N] += 1

    top_fraction = top_diff / (top_same + top_diff)
    qcd_fraction = qcd_diff / (qcd_same + qcd_diff)

    with open(f"/xai4hepvol/trained_top_fraction.pkl", 'wb') as f:
        pkl.dump(top_fraction, f)
    with open(f"/xai4hepvol/trained_qcd_fraction.pkl", 'wb') as f:
        pkl.dump(qcd_fraction, f)
