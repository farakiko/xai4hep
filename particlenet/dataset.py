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


class TopTaggingDataset(Dataset):
    """
    Initialize parameters of graph dataset
    Args:
        root (str): path
    """

    def __init__(self, root, mode, transform=None, pre_transform=None):
        super(TopTaggingDataset, self).__init__(root, transform, pre_transform)
        self.root = f"{self.root}/{mode}"
        self._processed_dir = Dataset.processed_dir.fget(self)
        self.mode = mode    # train/val/test

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, "*.pt"))
        return sorted([processed_path.replace(self.processed_dir, ".") for processed_path in proc_list])

    def prepare_ptfiles(self):
        """
        Loads the H5 file and splits it up into pt files for optimized PyTorch training/inference
        Transforms the input variables and stores only the following 7 features per particle
            1. ∆η difference in pseudorapidity between the particle and the jet axis
            2. ∆φ difference in azimuthal angle between the particle and the jet axis
            3. log pT logarithm of the particle’s pT
            4. log E logarithm of the particle’s energy
            5. log pT/pT_jet logarithm of the particle’s pT relative to the jet pT
            6. log E/E_jet logarithm of the particle’s energy relative to the jet energy
            7. ∆R
        """

        # https://github.com/jet-universe/particle_transformer/blob/main/utils/convert_top_datasets.py
        import os
        import pandas as pd
        import numpy as np
        import awkward0
        from uproot3_methods import TLorentzVectorArray
        import awkward as ak

        df = pd.read_hdf(f"{self.root}/raw/{self.mode}.h5", key='table')
        # df = dataframe.iloc[start:stop]

        def _col_list(prefix, max_particles=200):
            return ['%s_%d' % (prefix, i) for i in range(max_particles)]

        _px = df[_col_list('PX')].values
        _py = df[_col_list('PY')].values
        _pz = df[_col_list('PZ')].values
        _e = df[_col_list('E')].values

        mask = _e > 0
        n_particles = np.sum(mask, axis=1)

        px = awkward0.JaggedArray.fromcounts(n_particles, _px[mask])
        py = awkward0.JaggedArray.fromcounts(n_particles, _py[mask])
        pz = awkward0.JaggedArray.fromcounts(n_particles, _pz[mask])
        energy = awkward0.JaggedArray.fromcounts(n_particles, _e[mask])

        p4 = TLorentzVectorArray.from_cartesian(px, py, pz, energy)

        jet_p4 = p4.sum()

        # outputs
        v = {}
        v['label'] = df['is_signal_new'].values

        v['jet_pt'] = jet_p4.pt
        v['jet_eta'] = jet_p4.eta
        v['jet_phi'] = jet_p4.phi
        v['jet_energy'] = jet_p4.energy
        v['jet_mass'] = jet_p4.mass
        v['jet_nparticles'] = n_particles

        v['part_px'] = px
        v['part_py'] = py
        v['part_pz'] = pz
        v['part_energy'] = energy

        _jet_etasign = np.sign(v['jet_eta'])
        _jet_etasign[_jet_etasign == 0] = 1
        v['part_deta'] = (p4.eta - v['jet_eta']) * _jet_etasign
        v['part_dphi'] = p4.delta_phi(jet_p4)

        # https://github.com/jet-universe/particle_transformer/blob/main/data/TopLandscape/top_kin.yaml
        part_pt = np.hypot(v['part_px'], v['part_py'])
        part_pt_log = np.log(part_pt)
        part_e_log = np.log(v['part_energy'])
        part_logptrel = np.log(part_pt / v['jet_pt'])
        part_logerel = np.log(v['part_energy'] / v['jet_energy'])
        part_deltaR = np.hypot(v['part_deta'], v['part_dphi'])

        data = []
        c = 0
        for jet_index in range(len(df - 1)):

            data.append(
                Data(x=torch.cat([torch.from_numpy(v['part_deta'][jet_index].reshape(-1, 1)),
                                  torch.from_numpy(v['part_dphi'][jet_index].reshape(-1, 1)),
                                  torch.from_numpy(part_pt_log[jet_index].reshape(-1, 1)),
                                  torch.from_numpy(part_e_log[jet_index].reshape(-1, 1)),
                                  torch.from_numpy(part_logptrel[jet_index].reshape(-1, 1)),
                                  torch.from_numpy(part_logerel[jet_index].reshape(-1, 1)),
                                  torch.from_numpy(part_deltaR[jet_index].reshape(-1, 1))], axis=1),
                     y=torch.tensor(v['label'][jet_index]).long()))
            if jet_index % 100000 == 0 and jet_index != 0:
                print(f"saving datafile data_{c}")
                torch.save(data, f'{self.processed_dir}/data_{c}.pt')
                c += 1
                data = []

    def __len__(self):
        proc_list = glob(osp.join(self.processed_dir, "*.pt"))
        return len(sorted([processed_path.replace(self.processed_dir, ".") for processed_path in proc_list]))

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(idx)), map_location="cpu")
        return data

    def __getitem__(self, idx):
        return self.get(idx)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Input data path")
    parser.add_argument("--mode", type=str, required=True, help="'train' or 'val' or 'test'?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    """
    To process train data: python TopTaggingDataset.py --dataset ../data/toptagging/ --mode train
    To process val data:  python TopTaggingDataset.py --dataset ../data/toptagging/ --mode val
    """

    args = parse_args()

    topdataset = TopTaggingDataset(root=args.dataset, mode=args.mode)
    topdataset.prepare_ptfiles()
