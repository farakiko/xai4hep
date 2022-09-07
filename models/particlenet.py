import time
import matplotlib.pyplot as plt
import matplotlib
import torch_geometric
from torch_geometric.loader import DataListLoader, DataLoader
from jetnet.datasets import JetNet, TopTagging
import pandas as pd
import h5py
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

from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_cluster import knn_graph

import numpy as np
import torch.nn.functional as F

try:
    from torch_cluster import knn
except ImportError:
    knn = None


class EdgeConv_lrp(MessagePassing):
    """
    Copied from pytorch_geometric source code, with the following edits
    1. torch.cat([x_i, x_j - x_i], dim=-1)) -> torch.cat([x_i, x_j], dim=-1))
    2. retrieve edge_activations
    """

    def __init__(self, nn: Callable, aggr: str = "max", **kwargs):
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
        return f"{self.__class__.__name__}(nn={self.nn})"


class EdgeConvBlock(nn.Module):
    def __init__(self, in_size, layer_size):
        super(EdgeConvBlock, self).__init__()

        layers = []

        layers.append(nn.Linear(in_size * 2, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for i in range(1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.edge_conv = EdgeConv_lrp(nn.Sequential(*layers), aggr="mean")

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)


class ParticleNet(nn.Module):
    def __init__(self, node_feat_size, num_classes=5, k=3):
        super(ParticleNet, self).__init__()
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes

        self.k = k
        self.num_edge_conv_blocks = 3
        self.kernel_sizes = [self.node_feat_size, 64, 128, 256]
        self.input_sizes = np.cumsum(self.kernel_sizes)  # [4, 4+64, 4+64+128, 4+64+128+256]
        self.fc_size = 256
        # self.dropout = 0.1
        # self.dropout_layer = nn.Dropout(p=self.dropout)

        # define the edgeconvblocks
        self.edge_conv_blocks = nn.ModuleList()
        for i in range(0, self.num_edge_conv_blocks):
            self.edge_conv_blocks.append(EdgeConvBlock(self.input_sizes[i], self.kernel_sizes[i + 1]))

        # define the fully connected networks (post-edgeconvs)
        self.fc1 = nn.Linear(self.input_sizes[-1], self.fc_size)
        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

    def forward(self, batch, relu_activations=False):
        x = batch.x
        batch = batch.batch
        edge_activations = {}
        edge_block_activations = {}
        edge_index = {}

        for i in range(self.num_edge_conv_blocks):

            # using only angular coords for knn in first edgeconv block
            edge_index[f"edge_conv_{i}"] = knn_graph(x[:, :2], self.k, batch) if i == 0 else knn_graph(x, self.k, batch)

            out, edge_activations[f"edge_conv_{i}"] = self.edge_conv_blocks[i](x, edge_index[f"edge_conv_{i}"])

            x = torch.cat((out, x), dim=1)  # concatenating with latent features i.e. skip connections per EdgeConvBlock

            edge_block_activations[f"edge_conv_{i}"] = x

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        # x = self.dropout_layer(F.relu(x))
        x = self.fc2(x)

        # no softmax because pytorch cross entropy loss includes softmax
        return x, edge_activations, edge_block_activations, edge_index


def prepare_data(dataframe, start=0, stop=-1):
    """
    Transforming the input variables and returning only the following 7 features per particle
        1. ∆η difference in pseudorapidity between the particle and the jet axis
        2. ∆φ difference in azimuthal angle between the particle and the jet axis
        3. log pT logarithm of the particle’s pT
        4. log E logarithm of the particle’s energy
        5. log pT/pT_jet logarithm of the particle’s pT relative to the jet pT
        6. log E/E_jet logarithm of the particle’s energy relative to the jet energy
        7. ∆R

    Returns
        a list of Data() objects with x~(num_particles,7) and y~(label)
    """

    # https://github.com/jet-universe/particle_transformer/blob/main/utils/convert_top_datasets.py
    import os
    import pandas as pd
    import numpy as np
    import awkward0
    from uproot3_methods import TLorentzVectorArray
    import awkward as ak

    df = dataframe.iloc[start:stop]

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

    # for jet_index in range(len(df - 1)):
    for jet_index in range(1000):

        data.append(
            Data(x=torch.cat([torch.from_numpy(v['part_deta'][jet_index].reshape(-1, 1)),
                              torch.from_numpy(v['part_dphi'][jet_index].reshape(-1, 1)),
                              torch.from_numpy(part_pt_log[jet_index].reshape(-1, 1)),
                              torch.from_numpy(part_e_log[jet_index].reshape(-1, 1)),
                              torch.from_numpy(part_logptrel[jet_index].reshape(-1, 1)),
                              torch.from_numpy(part_logerel[jet_index].reshape(-1, 1)),
                              torch.from_numpy(part_deltaR[jet_index].reshape(-1, 1))], axis=1),
                 y=torch.tensor(v['label'][jet_index]).long()))

    return data


t0 = time.time()
print('Loading the h5 file...')
# dataframe = pd.read_hdf("../data/toptagging/train.h5", key='table')
dataframe = pd.read_hdf("/xai4hepvol/toptagging/train.h5", key='table')
print(f'{round((time.time()-t0)/60,3)}min')

t0 = time.time()
print('Building the dataloader...')
data = prepare_data(dataframe, start=0, stop=-1)
loader = DataLoader(data, batch_size=2, shuffle=True)
print(f'{round((time.time()-t0)/60,3)}min')

num_features = 7
model = ParticleNet(node_feat_size=num_features, num_classes=1)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
sig = nn.Sigmoid()

print('Initiating the training...')
losses_tot = []
for epoch in range(100):
    t0 = time.time()

    losses = []
    for batch in loader:

        preds, _, _, _ = model(batch)

        loss = criterion(sig(preds), batch.y.reshape(-1, 1).float())

        # backprop
        optimizer.zero_grad()  # To avoid accumulating the gradients
        loss.backward()
        optimizer.step()

        losses.append(loss.to('cpu').detach().numpy())

    losses_tot.append(sum(losses) / len(losses))

    fig, ax = plt.subplots()
    ax.plot(range(len(losses_tot)), losses_tot, label='train loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Classifier loss')
    ax.legend(loc='best')
    # plt.show()
    plt.savefig(f'/xai4hep/loss_epoch_{epoch}')
    # plt.savefig(f'./loss_epoch_{epoch}')

    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, f'/xai4hep/model_epoch_{epoch}.pth')
    # torch.save(state_dict, f'./model_epoch_{epoch}.pth')

    print(f'Epoch {epoch}: {round((time.time()-t0)/60,3)}min per epoch, avg_loss={losses_tot[epoch]}')

# # get sample dataset
# dataset = jetnet.datasets.JetNet(jet_type="g")
#
# # load the dataset in a convenient pyg format
# dataset_pyg = []
# for aa in dataset:
#     x = aa[0][aa[0][:, 3] == 0.5][:, :3]  # skip the mask
#     d = Data(x=x, y=data[1])
#     dataset_pyg.append(d)
#
# len(dataset_pyg)
# loader = DataLoader(dataset_pyg, batch_size=3, shuffle=False)
#
# for i, batch in enumerate(loader):
#     break
#
# batch
#
#
# model = ParticleNet(node_feat_size=3)
#
# # try:
# #     state_dict = model.module.state_dict()
# # except AttributeError:
# #     state_dict = model.state_dict()
# # torch.save(state_dict, f'../state_dict.pth')
#
#
# _, _, _, edge_index = model(batch)
# edge_index["edge_conv_0"]
