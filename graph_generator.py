import os
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops


class HomologyGraphStats(torch.nn.Module):
    """
    Statistic graph structure generator
    """
    def __init__(
            self, dataset: str, homology: str, knn_k: int, loop: bool, flow: str,
            force_undirected: bool, 
            only_from_train: bool, no_test_test: bool,
            train_mask: Tensor, valid_mask: Tensor, test_mask: Tensor
    ):
        super(HomologyGraphStats, self).__init__()

        self.dataset = dataset
        self.homology = homology
        self.knn_k = knn_k
        self.loop = loop
        self.flow = flow
        self.force_undirected = force_undirected
        self.only_from_train = only_from_train
        self.no_test_test = no_test_test
        self.train_index = torch.nonzero(train_mask).squeeze(-1)
        self.valid_index = torch.nonzero(valid_mask).squeeze(-1)
        self.test_index = torch.nonzero(test_mask).squeeze(-1)

    def forward(self, x: Tensor, logger, similarity_matrix=None):

        if self.homology == "knn_pyg":
            edge_index, edge_attr = knn_graph_pyg(
                x=x,
                k=self.knn_k, loop=self.loop,
                flow=self.flow, cosine=False,
                force_undirected=self.force_undirected
            )
        elif self.homology == "knn_fast":
            edge_index, edge_attr = knn_graph_fast(
                x=x, k=self.knn_k, loop=self.loop,
                flow=self.flow, cosine=False,
                force_undirected=self.force_undirected,
                only_from_train=self.only_from_train,
                no_test_test=self.no_test_test,
                train_index=self.train_index,
                valid_index=self.valid_index,
                test_index = self.test_index,
            )
        else:
            raise ValueError(("%s is not a valid graph option" % self.homology))

        return edge_index, edge_attr


def knn_graph_pyg(
        x: Tensor,
        k: int = 15, loop: bool = True,
        flow: str = "source_to_target",
        cosine: bool = False,
        num_workers: int = 1,
        force_undirected: bool = False
):
    r""" Code from PyG
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/knn_graph.html#KNNGraph
    Creates a k-NN graph based on node positions :obj:`data.pos`
    (functional name: :obj:`knn_graph`).

    Args:
        k (int, optional): The number of neighbors. (default: :obj:`6`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        force_undirected (bool, optional): If set to :obj:`True`, new edges
            will be undirected. (default: :obj:`False`)
        flow (str, optional): The flow direction when used in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`).
            If set to :obj:`"source_to_target"`, every target node will have
            exactly :math:`k` source nodes pointing to it.
            (default: :obj:`"source_to_target"`)
        cosine (bool, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
    """

    # data.edge_attr = None
    # batch = data.batch if 'batch' in data else None
    edge_attr = None
    batch = None
    # select k neighbour nodes except itself
    k = k+1 if loop else k

    edge_index = knn_graph(
        x,
        k,
        batch,
        loop=loop,
        flow=flow,
        cosine=cosine,
        num_workers=num_workers,
    )

    if force_undirected:
        # edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))

    # data.edge_index = edge_index

    return edge_index, edge_attr


def knn_graph_fast(
        x: Tensor,
        train_index: Tensor,
        valid_index: Tensor,
        test_index: Tensor,
        k: int = 15,
        flow: str = "source_to_target",
        cosine: bool = False,
        loop: bool = True,
        only_from_train: bool = False,
        no_test_test: bool = False,
        force_undirected: bool = False,
):
    r""" Code from SLAPS
        https://github.com/BorealisAI/SLAPS-GNN/blob/main/utils.py#L12

        Args:
            k (int, optional): The number of neighbors. (default: :obj:`6`)
            loop (bool, optional): If :obj:`True`, the graph will contain
                self-loops. (default: :obj:`False`)
            force_undirected (bool, optional): If set to :obj:`True`, new edges
                will be undirected. (default: :obj:`False`)
            flow (str, optional): The flow direction when used in combination with
                message passing (:obj:`"source_to_target"` or
                :obj:`"target_to_source"`).
                If set to :obj:`"source_to_target"`, every target node will have
                exactly :math:`k` source nodes pointing to it.
                (default: :obj:`"source_to_target"`)
            cosine (bool, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        """
    device = x.device

    x = F.normalize(x, dim=1, p=2)

    if cosine:
        similarities = F.cosine_similarity(
            x[:, :, None], x.t()[:, :, None]
        )
    else:
        similarities = torch.mm(x, x.t())

    edge_index, edge_attr = process_similarity_matrix(
        similarities=similarities, x=x, k=k,
        train_index=train_index, valid_index=valid_index, test_index=test_index,
        only_from_train=only_from_train, no_test_test=no_test_test,
        flow=flow, loop=loop, force_undirected=force_undirected, device=device
    )

    return edge_index, edge_attr


def process_similarity_matrix(
        similarities, x, k, train_index, valid_index, test_index,
        only_from_train, no_test_test,
        flow, loop, force_undirected, device
):
    similarities.fill_diagonal_(0.)

    assert not (only_from_train == no_test_test == True)

    if only_from_train:
        # only allow edges from train to valid/test nodes
        _index = torch.cat([valid_index, test_index])
        similarities[:, _index] = 0
    elif no_test_test:
        # avoid test nodes connect test nodes
        _diagonal = similarities[test_index, test_index]
        _replace = similarities[test_index, :]
        _replace[:, test_index] = 0
        similarities[test_index, :] = _replace
        similarities[test_index, test_index] = _diagonal
    else:
        pass

    vals, inds = torch.tensor(similarities).topk(k=k, dim=-1)

    cols = inds.view(-1).long().to(device)
    values = vals.view(-1).to(device)
    rows = torch.arange(x.size(0)).view(-1, 1).repeat(1, k).view(-1).long().to(device)

    if flow == "source_to_target":
        edge_index = torch.stack([cols, rows], dim=0)
    elif flow == "target_to_source":
        edge_index = torch.stack([rows, cols], dim=0)
    else:
        raise ValueError(("%s is not a valid flow" % flow))

    if only_from_train:
        edge_index, edge_attr = remove_self_loops(
            edge_index=edge_index, edge_attr=values
        )
        _index = torch.nonzero(~torch.isin(edge_index[1], train_index.to(device))).squeeze(-1)
        edge_index = edge_index[:, _index]
        edge_attr = edge_attr[_index]
    else:
        if loop:
            edge_index, edge_attr = add_self_loops(
                edge_index=edge_index, edge_attr=values
            )
        else:
            edge_index, edge_attr = edge_index, values

        if force_undirected:
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=x.size(0))

    del similarities, cols, rows, values, inds

    return edge_index, edge_attr
