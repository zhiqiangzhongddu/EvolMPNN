import gc
import os
from typing import Optional, Tuple, Union
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from residueformer import (
    ResidueFormerConv, softmax_kernel_transformation, kernelized_softmax
)
import torchdrug.models as tg_model
import torchdrug.data as td_data

from utils import onehot, slice_indices


my_llm_folder = "./input/model_parameters"
llm_folder = "./input/esm-model-weights"
feature_folder = "./input/protein_features"
url = {
    "ESM-1b": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
    "ESM-1v": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt",
    "ESM-1b-regression":
        "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
    "ESM-2-8M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
    "ESM-2-35M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
    "ESM-2-150M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
    "ESM-2-650M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
    "ESM-2-3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
    "ESM-2-15B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt",
}


class ResidueConv(MessagePassing):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: int = 1,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_x = Linear(in_channels=in_channels, out_channels=heads * out_channels,
                            bias=False, weight_initializer='glorot')
        self.lin_residue_diff = Linear(in_channels=in_channels, out_channels=heads * out_channels,
                                       bias=False, weight_initializer='glorot')
        self.lin_agg_heads = Linear(in_channels=heads * out_channels, out_channels=out_channels,
                                    bias=False, weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        # The learnable parameters to compute residue evolution feature
        self.att_residue_diff = Parameter(torch.empty(1, heads, out_channels))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        self.lin_x.reset_parameters()
        self.lin_residue_diff.reset_parameters()
        self.lin_agg_heads.reset_parameters()

        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_residue_diff)

        zeros(self.bias)

    def forward(
            self, x: Tensor, edge_index: Tensor, residue_diff: Tensor,
            size: Size = None, return_attention_weights=None
    ):
        # N: num_sequence, n: num_residue, d: in_channels
        H, C = self.heads, self.out_channels

        # We first transform the input node features.
        x_src = x_dst = self.lin_x(x).view(-1, H, C)
        x = (x_src, x_dst)

        # Then, we transform the residue features and compute edge-wise residue evolution
        residue_diff = self.lin_residue_diff(residue_diff).view(-1, H, C)

        # Next, we compute node-level attention coefficients,
        # for source & target nodes and residue revolution (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha_residue_diff = (residue_diff * self.att_residue_diff).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        # edge_updater_type: (alpha: OptPairTensor, residue_diff: Tensor)
        alpha = self.edge_updater(edge_index=edge_index, alpha=alpha, alpha_r=alpha_residue_diff)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index=edge_index, x=x, alpha=alpha, size=size)

        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_agg_heads(out)

        if self.bias is not None:
            out = out + self.bias

        return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor, alpha_r: Tensor,
                    index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j + alpha_r if alpha_i is None else alpha_j + alpha_i + alpha_r

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor):
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


# v1
class ResidueMPNN(torch.nn.Module):

    def __init__(
            self, in_channels: int, residue_dim: int, hidden_channels: int, out_channels: int,
            num_residue: int, num_sequences: int,
            num_gnn_layers: int = 2, num_former_layers: int = 2, num_heads: int = 4, dropout: float = 0.0,
            use_bn: bool = True, use_residual: bool = True, use_act: bool = False, use_jk: bool = False,
    ):
        super(ResidueMPNN, self).__init__()

        hidden_channels = in_channels if hidden_channels < 0 else hidden_channels

        self.in_channels = in_channels
        self.residue_dim = residue_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_sequences = num_sequences
        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.use_jk = use_jk

        self.lin_x_residue = nn.Linear(residue_dim, hidden_channels)
        self.lin_residue_diff = nn.Linear(hidden_channels, hidden_channels)
        self.residue_bn = nn.LayerNorm(hidden_channels)

        self.pos_embedding = nn.Parameter(torch.empty(num_residue, hidden_channels))

        self.lin_x_seq = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_gnn_layers):
            self.convs.append(
                ResidueConv(
                    in_channels=hidden_channels, out_channels=hidden_channels,
                    heads=num_heads, num_residue=num_residue, num_sequences=num_sequences
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        if use_jk:
            self.lin_out = nn.Linear(hidden_channels * num_gnn_layers + hidden_channels, out_channels)
        else:
            self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_seq, x_residue, edge_index, n_id:Optional = None):
        # N: num_sequence, n: num_residue, E: num_edge, d: in_channels
        N = n_id.size(0) if n_id is not None else x_seq.size(0)
        light = True if x_residue.dim() == 2 else False

        x_seq = x_seq[n_id].to(self.device) if n_id is not None else x_seq.to(self.device)
        x_residue = x_residue[n_id].to(self.device) if n_id is not None else x_residue.to(self.device)

        z_residue = self.lin_x_residue(x_residue)
        if not light:
            z_residue = z_residue * self.pos_embedding.repeat(N, 1, 1)

        z_residue_src = z_residue[edge_index[0]]
        z_residue_dst = z_residue[edge_index[1]]
        if not light:
            residue_diff = (z_residue_dst - z_residue_src).sum(1)  # E * d
        else:
            residue_diff = (z_residue_dst - z_residue_src)  # E * d
        residue_diff = self.lin_residue_diff(residue_diff)

        _layer = []

        z = self.lin_x_seq(x_seq)
        if self.use_bn:
            z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        _layer.append(z)

        for i, conv in enumerate(self.convs):
            z = conv(x=z, edge_index=edge_index, residue_diff=residue_diff)

            if self.use_residual:
                z += _layer[i]
            if self.use_bn:
                z = self.bns[i + 1](z)
            if self.use_act:
                z = self.activation(z)

            z = F.dropout(z, p=self.dropout, training=self.training)
            _layer.append(z[:, :conv.out_channels])

        if self.use_jk: # use jk connection for each layer
            z = torch.cat(_layer, dim=-1)

        out = self.lin_out(z).squeeze(0)

        return out

