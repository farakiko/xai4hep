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


class FFN(nn.Module):
    """
    Showcase an example of an FFN model, with a skip connection, that can be explained by LRP
    """

    def __init__(self, input_dim=3, hidden_dim=256, embedding_dim=40, output_dim=2):
        super(FFN, self).__init__()

        self.act = nn.ReLU

        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X):
        embedding = self.nn1(X)
        return self.nn2(torch.cat([X, embedding], axis=1))
