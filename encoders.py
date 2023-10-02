from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor


class GCNEncoder(torch.nn.Module):
    def __init__(
            self, n_layers: int,
            in_channels: int, hidden_channels: int = 256, out_channels: int = 0,
            activation: str ="relu", dropout: float = 0.5
    ):
        super(GCNEncoder, self).__init__()
        self.num_layers = n_layers
        self.activation = getattr(F, activation)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        out_channels = hidden_channels if out_channels < 1 else out_channels

        self.layers = nn.ModuleList()
        for i in range(n_layers-1):
            if i == 0:
                self.layers.append(GCNConv(
                    in_channels=in_channels, out_channels=hidden_channels, cached=False
                ))
            else:
                self.layers.append(GCNConv(
                    in_channels=hidden_channels, out_channels=hidden_channels, cached=False
                ))
        self.layers.append(GCNConv(
            in_channels=hidden_channels, out_channels=out_channels, cached=False
        ))

    def forward(self, x, edge_index, edge_weight=None):
        if self.dropout:
            x = self.dropout(x)

        for layer in self.layers[:-1]:
            x = layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x


class GATEncoder(torch.nn.Module):

    def __init__(
            self, n_layers: int,
            in_channels: int, hidden_channels: int = 256, out_channels: int = 0,
            activation: str ="relu", dropout: float = 0.5, heads: int = 8,
    ):
        super(GATEncoder, self).__init__()
        self.num_layers = n_layers
        self.activation = getattr(F, activation)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        out_channels = hidden_channels if out_channels < 1 else out_channels

        self.layers = nn.ModuleList()
        for i in range(n_layers-1):
            if i == 0:
                self.layers.append(GATConv(
                    in_channels=in_channels, out_channels=hidden_channels,
                    heads=heads, dropout=dropout
                ))
            else:
                self.layers.append(GATConv(
                    in_channels=hidden_channels * heads, out_channels=hidden_channels,
                    heads=heads, dropout=dropout
                ))
        self.layers.append(GATConv(
            in_channels=hidden_channels * heads, out_channels=out_channels,
            heads=1, concat=False, dropout=dropout
        ))

    def forward(self, x, edge_index, edge_weight=None):

        if self.dropout:
            x = self.dropout(x)

        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x, edge_index, edge_weight)

        return x


class TransformerEncoder(torch.nn.Module):

    def __init__(
            self, n_layers: int,
            in_channels: int, hidden_channels: int = 256, out_channels: int = 0,
            activation: str ="relu", dropout: float = 0.5, heads: int = 8, beta: bool = False
    ):
        super(TransformerEncoder, self).__init__()
        self.num_layers = n_layers
        self.activation = getattr(F, activation)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        out_channels = hidden_channels if out_channels < 1 else out_channels

        self.layers = nn.ModuleList()
        for i in range(n_layers-1):
            if i == 0:
                self.layers.append(TransformerConv(
                    in_channels=in_channels, out_channels=hidden_channels,
                    heads=heads, dropout=dropout, beta=beta
                ))
            else:
                self.layers.append(TransformerConv(
                    in_channels=hidden_channels * heads, out_channels=hidden_channels,
                    heads=heads, dropout=dropout, beta=beta
                ))
        self.layers.append(TransformerConv(
            in_channels=hidden_channels * heads, out_channels=out_channels,
            heads=1, concat=False, dropout=dropout, beta=beta
        ))

    def forward(self, x: Tensor, edge_index: Adj, edge_weight=None):

        if self.dropout:
            x = self.dropout(x)

        for layer in self.layers[:-1]:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_weight)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_weight)

        return x
