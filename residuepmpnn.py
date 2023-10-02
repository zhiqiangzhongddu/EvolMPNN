import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchdrug.models as tg_model

from residueformer import (
    ResidueFormerConv, softmax_kernel_transformation, kernelized_softmax
)


# v-5 use series pipeline
class ResiduePMPNNConv(nn.Module):
    def __init__(
            self,
            hid_channels: int,
            operator: str = 'mean',
    ):
        super(ResiduePMPNNConv, self).__init__()
        self.operator = operator

        # self.lin_dist = nn.Linear(hid_channels, hid_channels)
        self.lin_z_seq = nn.Linear(hid_channels + hid_channels, hid_channels)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # self.lin_dist.reset_parameters()
    #     self.lin_z_seq.reset_parameters()

    def forward(self, x_seq: Tensor, dists: Tensor, num_anchor: int):

        anchor_seq = x_seq[num_anchor, :].unsqueeze(0).repeat(x_seq.size(0), 1, 1)
        messages = dists * anchor_seq
        if self.operator == "mean":
            messages = torch.mean(messages, dim=1) # (N+a)*d
        elif self.operator == 'sum':
            messages = torch.sum(messages, dim=1) # (N+a)*d

        z_seq = torch.cat((x_seq, messages), dim=1)  # if x_seq has no anchor, messages only take first N
        z_seq = self.lin_z_seq(z_seq)

        return z_seq


class ResiduePMPNN(torch.nn.Module):
    def __init__(
            self, in_channels: int, residue_dim: int, hidden_channels: int, out_channels: int,
            num_residue: int, num_sequences: int,
            num_gnn_layers: int = 2, num_former_layers: int = 2, dropout: float = 0.0, num_anchor=-1,
            pos_embed=True,
            use_bn: bool = True, use_residual: bool = True, use_act: bool = False, use_jk: bool = False,
            kernel_transformation=softmax_kernel_transformation, nb_random_features=30, num_heads=4,
            use_gumbel=True, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid',
            use_edge_loss=False,
    ):
        super(ResiduePMPNN, self).__init__()

        hidden_channels = in_channels if hidden_channels < 0 else hidden_channels

        self.in_channels = in_channels
        self.residue_dim = residue_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_gnn_layers = num_gnn_layers
        self.num_former_layers = num_former_layers
        self.num_sequences = num_sequences
        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.num_anchor = num_anchor
        self.anchors = []

        self.lin_x_residue = nn.Linear(residue_dim, hidden_channels)
        self.pos_embedding = nn.Parameter(torch.empty(num_residue, hidden_channels))
        nn.init.kaiming_normal_(self.pos_embedding, nonlinearity='relu')

        self.lin_x_seq = nn.Linear(in_channels, hidden_channels)

        self.residue_formers = nn.ModuleList()
        self.residue_bns = nn.ModuleList()
        self.residue_bns.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_former_layers):
            self.residue_formers.append(
                ResidueFormerConv(
                    in_channels=hidden_channels, out_channels=hidden_channels, num_heads=num_heads,
                    kernel_transformation=kernel_transformation,
                    nb_random_features=nb_random_features,
                    use_gumbel=use_gumbel, nb_gumbel_sample=nb_gumbel_sample,
                    rb_order=rb_order, rb_trans=rb_trans, use_edge_loss=use_edge_loss
                )
            )
            self.residue_bns.append(nn.LayerNorm(hidden_channels))

        self.seq_bns = nn.ModuleList()
        self.seq_bns.append(nn.LayerNorm(hidden_channels))
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(
                ResiduePMPNNConv(
                    hid_channels=hidden_channels, operator="mean",
                )
            )
            self.seq_bns.append(nn.LayerNorm(hidden_channels))

        self.lin_out = nn.Linear(hidden_channels * 2, out_channels)

        self.preselect_anchor()

    def preselect_anchor(self):

        self.num_anchor = int(np.log2(self.num_sequences)) if self.num_anchor < 0 else self.num_anchor
        self.anchors = torch.LongTensor(np.random.choice(range(self.num_sequences), self.num_anchor))

    def residue_diff(self, z_residue):

        if z_residue.size(0) == 1:
            z_residue = z_residue.squeeze(0)
            anchor_residue = z_residue[-self.num_anchor:, :]
            dists = z_residue.unsqueeze(1).repeat(1, self.num_anchor, 1) - \
                            anchor_residue.unsqueeze(0).repeat(z_residue.size(0), 1, 1)
        else:
            anchor_residue = z_residue[-self.num_anchor:, :]
            full_dists = z_residue.unsqueeze(1).repeat(1, self.num_anchor, 1, 1) - \
                    anchor_residue.unsqueeze(0).repeat(z_residue.size(0), 1, 1, 1)
            dists = full_dists.mean(2).mean(1)
        return dists

    def forward(self, x_seq, x_residue, tau, n_id: Optional = None):
        # N: num_sequence, n: num_residue, a: num_anchor, E: num_edge, d: in_channels
        N = n_id.size(0) if n_id is not None else x_seq.size(0)
        light = True if x_residue.dim() == 2 else False

        # main dataset
        _residue = x_residue.to(self.device) if n_id is None else x_residue[n_id].to(self.device)
        _seq = x_seq.to(self.device) if n_id is None else x_seq[n_id].to(self.device)
        # anchor
        anchor_residue = x_residue[self.anchors].to(self.device)
        anchor_seq = x_seq[self.anchors].to(self.device)
        # get seq and residue data
        x_residue = torch.cat([_residue, anchor_residue], dim=0)
        x_seq = torch.cat([_seq, anchor_seq], dim=0)

        # residue channel
        _layer_residue = []
        z_residue = self.lin_x_residue(x_residue)
        if not light:
            z_residue = z_residue * self.pos_embedding.repeat(z_residue.size(0), 1, 1)
        else:
            z_residue = z_residue.unsqueeze(0)
        if self.use_bn:
            z_residue = self.residue_bns[0](z_residue)
        if self.use_act:
            z_residue = self.activation(z_residue)
        z_residue = F.dropout(z_residue, p=self.dropout, training=self.training)
        _layer_residue.append(z_residue)
        # apply R-R layers
        for i in range(self.num_former_layers):
            z_residue = self.residue_formers[i](
                z=z_residue, adjs=[None], tau=tau
            )
            if self.use_residual:
                z_residue += _layer_residue[i]
            if self.use_bn:
                z_residue = self.residue_bns[i + 1](z_residue)
            if self.use_act:
                z_residue = self.activation(z_residue)
            z_residue = F.dropout(z_residue, p=self.dropout, training=self.training)
            _layer_residue.append(z_residue)
        # (N + a) * a * d
        dists_residue = self.residue_diff(z_residue=z_residue)
        if not light:
            z_residue = z_residue.mean(1)
        else:
            z_residue = z_residue.squeeze(0)

        # seq channel
        _layer_seq = []
        z_seq = self.lin_x_seq(x_seq)
        if self.use_bn:
            z_seq = self.seq_bns[0](z_seq)
        if self.use_act:
            z_seq = self.activation(z_seq)
        z_seq = F.dropout(z_seq, p=self.dropout, training=self.training)
        _layer_seq.append(z_seq)
        # apply Seq layer
        for i in range(self.num_gnn_layers):
            z_seq = self.convs[i](
                x_seq=z_seq, dists=dists_residue, num_anchor=self.num_anchor
            )
            if self.use_residual:
                z_seq += _layer_seq[i]
            if self.use_bn:
                z_seq = self.seq_bns[i + 1](z_seq)
            if self.use_act:
                z_seq = self.activation(z_seq)
            z_seq = F.dropout(z_seq, p=self.dropout, training=self.training)
            _layer_seq.append(z_seq)

        # combination
        z = torch.cat([z_seq[:N], z_residue[:N]], dim=1)
        out = self.lin_out(z).squeeze(0)

        return out
