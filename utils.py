import os
import random
import numpy as np
import pandas as pd
import argparse
from IPython import get_ipython
import logging
from datetime import datetime

import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max


def parse_args():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--gpu_id", help="device", type=str, default="0")

    # dataset
    parser.add_argument("--dataset", help="dataset name", type=str, default="GB1",
                        choices=["GB1", "Fluorescence", "AAV"])
    parser.add_argument("--split", help="data split setting", type=str, default="two_vs_rest")
    parser.add_argument('--batch_size', help="batch size", type=int, default=256)
    parser.add_argument("--load_protein_feat", help="if load protein feature", type=str2bool,
                        default=True)
    parser.add_argument("--feature_generator", help="node/residue feature generator", type=str, default="ESM-1b",
                        choices=["ESM-1b", "ESM-1v", "ESM-2-150M", "ESM-2-650M", 'CNN', 'BERT', 'ProtBert'])
    parser.add_argument("--fine_tuned_generator", help="if generate node features use fine-tuned LLM", type=str2bool,
                        default=True)
    parser.add_argument("--oh_residue_feat", help="if generate OneHot residue features", type=str2bool,
                        default=False)
    parser.add_argument("--full_residue_feat", help="if generate full residue features", type=str2bool,
                        default=False)
    parser.add_argument("--light_residue_feat", help="if generate light residue features", type=str2bool,
                        default=False)

    # homology network
    parser.add_argument("--homology", help="method to initialise homology network", type=str, default='knn_fast')
    parser.add_argument("--knn_k", help="K for KNN graph generation", type=int, default=10)
    parser.add_argument("--loop", help="generate graph with additional self-loop", type=str2bool, default=True)
    parser.add_argument("--flow", help="flow of generated graph", type=str, default="source_to_target")
    parser.add_argument("--force_undirected", help="if make generated graph undirected", type=str2bool, default=False)
    parser.add_argument("--no_test_test", help="if avoid connections between test and test nodes", type=str2bool,
                        default=False)
    parser.add_argument("--only_from_train", help="if only allow edges from train to valid/test", type=str2bool,
                        default=False)

    # model
    parser.add_argument("--model", help="model architecture", type=str,
                        choices=["Mlp", "Gnn", "LlmMlp", 'Transformer', 'NodeFormer',
                                 'ResidueMPNN', 'ResidueCnnMPNN', 'ResiduePMPNN', 'ResidueCnnPMPNN',
                                 'ResidueFormer', 'ResidueCnnFormer', 'ResiduePFormer', 'ResidueCnnPFormer'])
    parser.add_argument("--encoder", help="Encoder", type=str, default="ESM-1b",
                        choices=["ESM-1b", "ESM-1v", "ESM-2-150M", "ESM-2-650M", 'CNN', 'BERT', 'ProtBert'])
    parser.add_argument("--pretrained_encoder", help="if load pretrained ENCODER model parameters", type=str2bool,
                        default=True)
    parser.add_argument("--fix_encoder", help="if fix ENCODER model parameters", type=str2bool,
                        default=True)
    parser.add_argument("--gnn", help="GNN Encoder", type=str, default="GCN",
                        choices=['GCN', "GAT"])
    parser.add_argument("--gnn_hid_channels", help="hidden dimension of GNN encoder", type=int, default=128)
    parser.add_argument("--gnn_layers", help="number of layers of GNN encoder", type=int, default=2)
    parser.add_argument("--former_layers", help="number of layers of Former encoder", type=int, default=2)
    parser.add_argument("--num_heads", help="number of heads of GNN attention component", type=int, default=8)
    parser.add_argument("--gnn_dropout", help="dropout ratio", type=float, default=0.5)
    parser.add_argument("--mlp_layers", help="number of layers of MLP encoder", type=int, default=2)
    parser.add_argument("--mlp_hid_channels", help="hidden dimension of MLP", type=int, default=-1)
    parser.add_argument("--mlp_dropout", help="dropout ratio", type=float, default=0)
    parser.add_argument("--pretrained_mlp", help="if load pretrained MLP model parameters", type=str2bool,
                        default=False)
    parser.add_argument("--fix_mlp", help="if fix MLP model parameters", type=str2bool, default=False)
    parser.add_argument("--use_edge_attr", help="if use edge attr", type=str2bool, default=False)
    parser.add_argument("--use_edge_loss", help="if use edge loss", type=str2bool, default=False)
    parser.add_argument("--use_act", help="if use Activation function", type=str2bool, default=True)
    parser.add_argument("--use_bn", help="if use LayerNorm", type=str2bool, default=True)
    parser.add_argument("--use_residual", help="if use Residual", type=str2bool, default=True)
    parser.add_argument("--use_jk", help="if use JK", type=str2bool, default=False)
    parser.add_argument('--tau', type=float, default=1., help='temperature for gumbel softmax')
    parser.add_argument('--rb_order', type=int, default=0, help='order for relational bias, 0 for not use')
    parser.add_argument('--num_anchor', type=int, default=-1, help='number of anchor protein sequences')

    # train and evaluate
    parser.add_argument("--optimizer", help="optimizer", type=str, default="Adam",
                        choices=['RMSProp', 'Adam'])
    parser.add_argument("--epochs", help="number of epochs", type=int, default=100)
    # parser.add_argument("--adj_epochs", help="number of epochs to train adj learner", type=int, default=20)
    # parser.add_argument("--lamda_rec", help="weight of adj reconstruction loss", type=float, default=10.)
    parser.add_argument("--lamda_edge", help="weight of edge loss", type=float, default=0.1)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=5e-3)
    parser.add_argument("--lr_ratio", help="Llm model has smaller learning rate", type=float, default=1e-1)
    # parser.add_argument('--exponential_decay_step', type=int, default=5)
    # parser.add_argument('--decay_rate', type=float, default=0.5)

    return parser.parse_known_args()[0]


def is_notebook() -> bool:

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    except NameError:
        return False      # Probably standard Python interpreter


def spearmanr(pred, target):
    """
    Code: https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/metrics/metric.py#L370
    Spearman correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """

    def get_ranking(input):
        input_set, input_inverse = input.unique(return_inverse=True)
        order = input_inverse.argsort()
        ranking = torch.zeros(len(input_inverse), device=input.device)
        ranking[order] = torch.arange(1, len(input) + 1, dtype=torch.float, device=input.device)

        # for elements that have the same value, replace their rankings with the mean of their rankings
        mean_ranking = scatter_mean(ranking, input_inverse, dim=0, dim_size=len(input_set))
        ranking = mean_ranking[input_inverse]
        return ranking

    pred = get_ranking(pred)
    target = get_ranking(target)
    covariance = (pred * target).mean() - pred.mean() * target.mean()
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    spearmanr = covariance / (pred_std * target_std + 1e-10)

    return spearmanr


def set_seed(seed: int = 42):

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def slice_indices(indices, batch_size):

    sliced_lists = []
    for i in range(0, len(indices), batch_size):
        sliced_lists.append(indices[i:i + batch_size])

    return sliced_lists


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_edge(edge_index, fr, to):

    edge_index_fr = edge_index[0].cpu()
    common_elements_fr = torch.tensor(list(set(edge_index_fr.tolist()) & set(fr.cpu().tolist())))
    indices_fr = torch.nonzero(torch.isin(edge_index_fr, common_elements_fr)).squeeze()

    edge_index_to = edge_index[1].cpu()[indices_fr].cpu()
    common_elements = torch.tensor(list(set(edge_index_to.tolist()) & set(to.cpu().tolist())))
    indices = torch.nonzero(torch.isin(edge_index_to, common_elements)).squeeze()

    if indices_fr.size(0) == 0:
        return indices, indices.size(0), 0
    else:
        return indices, indices.size(0), indices.size(0)/indices_fr.size(0)


def check_adjmatrix(data, edge_index = None):

    if edge_index is None:
        edge_index = data.edge_index

    matrix = []
    for fr in ["train", "valid", "test"]:
        _row = []
        for to in ["train", "valid", "test"]:
            mask_fr, mask_to = fr + "_mask", to + "_mask"
            _, number, ratio = count_edge(
                edge_index=edge_index,
                fr=torch.nonzero(data[mask_fr]).squeeze(),
                to=torch.nonzero(data[mask_to]).squeeze()
            )
            _row.append(number)
        matrix.append(_row)

    return np.array(matrix)


def make_output_folder(args):

    folder = "../output/{}/{}_{}".format(
        args.dataset, args.model, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    os.makedirs(folder)
    return folder


def get_root_logger(folder):

    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if folder:
        handler = logging.FileHandler(os.path.join(folder, "log.txt"))
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger

def get_random_mask(features, r: int = 10, nr: int = 5):
    """
    Args:
        features:
        r: ratio of zeros to ones to mask out for binary features
        nr: ratio of ones to mask out for binary features and ratio of features to mask out for real values features
    Returns:

    """
    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    pzeros = nones / nzeros / r * nr

    probs = torch.zeros(features.shape).cuda()
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r

    mask = torch.bernoulli(probs)

    return mask

def adj_mul(adj_i, adj, N):

    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()

    return adj_j


def onehot(x, vocab):

    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1

    feature = [0] * len(vocab)
    if index == -1:
        raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
    feature[index] = 1

    return feature


def sample_node(data, sequences, num_node):
    # num_node = 1000
    data.x = data.x[:num_node]
    data.y = data.y[:num_node]
    data.train_mask = data.train_mask[:num_node]
    data.valid_mask = data.valid_mask[:num_node]
    data.test_mask = data.test_mask[:num_node]
    data.num_nodes = num_node
    _edge_index = data.edge_index[:, torch.nonzero(torch.isin(data.edge_index[0], torch.arange(num_node))).squeeze()]
    data.edge_index = _edge_index[:, torch.nonzero(torch.isin(_edge_index[1], torch.arange(num_node))).squeeze()]
    data.edge_attr = None
    sequences = sequences[:num_node]

    return data, sequences


def save_info(args, folder, result):
    data = args.dataset
    split = args.split
    model = args.model
    date = folder.split("_")[-1]

    RES_FILE = '../output/res_%s.csv' % data
    df_res = pd.DataFrame.from_dict(
        {'model': [model],
         'split': [split],
         'date': [date],
         'performance': [result],
         }
    )
    if os.path.exists(RES_FILE):
        df_res.to_csv(RES_FILE, mode='a', header=False, index=False)
    else:
        df_res.to_csv(RES_FILE, header=True, index=False)
    