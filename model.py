import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torchdrug.models as tg_model

from encoders import GCNEncoder, GATEncoder, TransformerEncoder
from graph_generator import knn_graph_fast
from protbert import ProtBert


esm_folder = "./input/esm-model-weights"
protbert_folder = "./input/protbert-model-weights/"


class MultiLayerPerceptron(nn.Module):
    def __init__(
            self, in_channels: int, hid_channels: int, out_channels: int,
            n_layers: int = 3, dropout: float = 0.5, activation: str ="relu",
    ):
        super(MultiLayerPerceptron, self).__init__()
        self.n_layers = n_layers
        self.activation = getattr(F, activation)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        hid_channels = in_channels if hid_channels < 0 else hid_channels

        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hid_channels))
            for i in range(n_layers-2):
                self.layers.append(nn.Linear(hid_channels, hid_channels))
            self.layers.append(nn.Linear(hid_channels, out_channels))

    def forward(self, data, proteins):

        if torch.is_tensor(data):
            x = data
        else:
            x = data.x

        if self.n_layers == 1:
            x = self.layers[0](x)
        else:
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
            x = self.layers[-1](x)

        if torch.is_tensor(data):
            return x, None, None
        else:
            return x, data.edge_index, data.edge_attr


class Gnn(nn.Module):

    def __init__(
            self, args, in_channels: int, out_channels: int,
            gnn_layers: int = 2, gnn_hid_channels: int = -1, gnn_dropout: float = 0.5, beta: bool = True
    ):
        super(Gnn, self).__init__()
        self.use_edge_attr = args.use_edge_attr

        gnn_hid_channels = in_channels if gnn_hid_channels < 0 else gnn_hid_channels

        if args.gnn == "GCN":
            self.gnn = GCNEncoder(
                in_channels=in_channels, hidden_channels=gnn_hid_channels, out_channels=out_channels,
                n_layers=gnn_layers, dropout=gnn_dropout
            )
        elif args.gnn == "GAT":
            self.gnn = GATEncoder(
                in_channels=in_channels, hidden_channels=gnn_hid_channels, out_channels=out_channels,
                n_layers=gnn_layers, dropout=gnn_dropout, heads=8
            )
        else:
            raise ValueError("%s is not a valid gnn option." % args.gnn)

    def forward(self, data, proteins):

        if self.use_edge_attr:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            x, edge_index, edge_attr = data.x, data.edge_index, None

        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)

        return x, None, None


class GnnMlp(nn.Module):

    def __init__(
            self, args, in_channels: int, out_channels: int,
            gnn_layers: int = 2, gnn_hid_channels: int = -1, gnn_dropout: float = 0.5, beta: bool = True,
            mlp_layers: int = 2, mlp_hid_channels: int = -1, mlp_dropout: float = 0.5
    ):
        super(GnnMlp, self).__init__()
        self.use_edge_attr = args.use_edge_attr

        gnn_hid_channels = in_channels if gnn_hid_channels < 0 else gnn_hid_channels
        mlp_hid_channels = in_channels if mlp_hid_channels < 0 else mlp_hid_channels

        if args.gnn == "GCN":
            self.gnn = GCNEncoder(
                in_channels=in_channels, hidden_channels=gnn_hid_channels,
                n_layers=gnn_layers, dropout=gnn_dropout
            )
        elif args.gnn == "GAT":
            self.gnn = GATEncoder(
                in_channels=in_channels, hidden_channels=gnn_hid_channels,
                n_layers=gnn_layers, dropout=gnn_dropout, heads=8
            )
        else:
            raise ValueError("%s is not a valid gnn option." % args.gnn)
        self.mlp = MultiLayerPerceptron(
            in_channels=gnn_hid_channels, hid_channels=mlp_hid_channels, out_channels=out_channels,
            n_layers=mlp_layers, dropout=mlp_dropout
        )

    def forward(self, data, proteins=None):

        if self.use_edge_attr:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            x, edge_index, edge_attr = data.x, data.edge_index, None

        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x, _, _ = self.mlp(data=x, proteins=proteins)

        return x, None, None


class LlmMlp(nn.Module):

    url = {
        "ESM-1b": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
        "ESM-1v": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt",
        # "ESM-1b-regression":
        #     "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
        # "ESM-2-8M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        # "ESM-2-35M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "ESM-2-150M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "ESM-2-650M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "ESM-2-3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
        # "ESM-2-15B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt",
    }

    def __init__(
            self, args, path=esm_folder, llm_model="ESM-1b", llm_readout="mean",
            mlp_layers: int = 2, mlp_hid_channels: int = -1, mlp_dropout: float = 0.5, out_channels: int = 1
    ):
        super(LlmMlp, self).__init__()

        assert llm_model in self.url.keys()
        self.encoder = tg_model.EvolutionaryScaleModeling(path=path, model=llm_model, readout=llm_readout)

        mlp_hid_channels = self.encoder.output_dim if mlp_hid_channels < 0 else mlp_hid_channels
        self.mlp = MultiLayerPerceptron(
            in_channels=self.encoder.output_dim, hid_channels=mlp_hid_channels, out_channels=out_channels,
            n_layers=mlp_layers, dropout=mlp_dropout
        )

    def forward(self, data, proteins):

        x = self.encoder(
            graph=proteins, input=proteins.node_feature.float(), all_loss=None, metric=None
        )["graph_feature"]
        x, _, _ = self.mlp(data=x, proteins=None)

        return x, None, None


class LlmGnnMlp(nn.Module):

    url = {
        "ESM-1b": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
        "ESM-1v": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt",
        # "ESM-1b-regression":
        #     "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
        # "ESM-2-8M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        # "ESM-2-35M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "ESM-2-150M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "ESM-2-650M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "ESM-2-3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
        # "ESM-2-15B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt",
    }

    def __init__(
            self, args, path=esm_folder, llm_model="ESM-1b", llm_readout="mean", out_channels: int = 1,
            gnn_layers: int = 2, gnn_hid_channels: int = -1, gnn_dropout: float = 0.5, beta: bool = True,
            mlp_layers: int = 2, mlp_hid_channels: int = -1, mlp_dropout: float = 0.5
    ):
        super(LlmGnnMlp, self).__init__()

        assert llm_model in self.url.keys()
        self.encoder = tg_model.EvolutionaryScaleModeling(path=path, model=llm_model, readout=llm_readout)

        self.use_edge_attr = args.use_edge_attr

        gnn_hid_channels = self.encoder.output_dim if gnn_hid_channels < 0 else gnn_hid_channels
        mlp_hid_channels = self.encoder.output_dim if mlp_hid_channels < 0 else mlp_hid_channels

        if args.gnn == "GCN":
            self.gnn = GCNEncoder(
                in_channels=self.encoder.output_dim, hidden_channels=gnn_hid_channels,
                n_layers=gnn_layers, dropout=gnn_dropout
            )
        elif args.gnn == "GAT":
            self.gnn = GATEncoder(
                in_channels=self.encoder.output_dim, hidden_channels=gnn_hid_channels,
                n_layers=gnn_layers, dropout=gnn_dropout, heads=8
            )
        else:
            raise ValueError("%s is not a valid gnn option." % args.gnn)
        self.mlp = MultiLayerPerceptron(
            in_channels=gnn_hid_channels, hid_channels=mlp_hid_channels, out_channels=out_channels,
            n_layers=mlp_layers, dropout=mlp_dropout
        )

    def forward(self, data, proteins):

        x = self.encoder(
            graph=proteins, input=proteins.node_feature.float(), all_loss=None, metric=None
        )["graph_feature"]
        edge_index = data.edge_index
        edge_attr = data.edge_attr if self.use_edge_attr else None

        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)
        new_edge_index, new_edge_attr = self.generate_graph(x=x.detach(), stop_region=data["test_mask"])
        x, _, _ = self.mlp(x)

        return x, new_edge_index, new_edge_attr


class CnnMlp(nn.Module):

    def __init__(
            self, args, input_dim: int = 21, hidden_dims = [1024, 1024], kernel_size = 5, padding = 2,
            mlp_layers: int = 2, mlp_hid_channels: int = -1, mlp_dropout: float = 0.5, out_channels: int = 1
    ):
        super(CnnMlp, self).__init__()

        self.encoder = tg_model.ProteinConvolutionalNetwork(
            input_dim=input_dim, hidden_dims=hidden_dims, kernel_size=kernel_size, padding=padding
        )

        mlp_hid_channels = hidden_dims[-1] if mlp_hid_channels < 0 else mlp_hid_channels
        self.mlp = MultiLayerPerceptron(
            in_channels=hidden_dims[-1], hid_channels=mlp_hid_channels, out_channels=out_channels,
            n_layers=mlp_layers, dropout=mlp_dropout
        )

    def forward(self, data, proteins):

        x = self.encoder(
            graph=proteins, input=proteins.node_feature.float(), all_loss=None, metric=None
        )["graph_feature"]
        x, _, _ = self.mlp(data=x, proteins=None)

        return x, None, None

class BertMlp(nn.Module):

    def __init__(
            self, args, input_dim = 21, hidden_dim = 512, num_layers = 4, num_heads = 8,
            intermediate_dim = 2048, hidden_dropout = 0.1, attention_dropout = 0.1,
            mlp_layers: int = 2, mlp_hid_channels: int = -1, mlp_dropout: float = 0.5, out_channels: int = 1
    ):
        super(BertMlp, self).__init__()

        self.encoder = tg_model.ProteinBERT(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            intermediate_dim=intermediate_dim, hidden_dropout=hidden_dropout, attention_dropout=attention_dropout,
        )

        mlp_hid_channels = hidden_dim if mlp_hid_channels < 0 else mlp_hid_channels
        self.mlp = MultiLayerPerceptron(
            in_channels=hidden_dim, hid_channels=mlp_hid_channels, out_channels=out_channels,
            n_layers=mlp_layers, dropout=mlp_dropout
        )

    def forward(self, data, proteins):

        x = self.encoder(
            graph=proteins, input=proteins.node_feature.float(), all_loss=None, metric=None
        )["graph_feature"]
        x, _, _ = self.mlp(data=x, proteins=None)

        return x, None, None


class ProtBertMlp(nn.Module):

    def __init__(
            self, args,
            mlp_layers: int = 2, mlp_hid_channels: int = -1, mlp_dropout: float = 0.5, out_channels: int = 1
    ):
        super(ProtBertMlp, self).__init__()

        self.encoder = ProtBert(path=protbert_folder)

        mlp_hid_channels = self.encoder.output_dim if mlp_hid_channels < 0 else mlp_hid_channels
        self.mlp = MultiLayerPerceptron(
            in_channels=self.encoder.output_dim, hid_channels=mlp_hid_channels, out_channels=out_channels,
            n_layers=mlp_layers, dropout=mlp_dropout
        )

    def forward(self, data, proteins):

        x = self.encoder(
            graph=proteins, input=proteins.node_feature.float(), all_loss=None, metric=None
        )["graph_feature"]
        x, _, _ = self.mlp(data=x, proteins=None)

        return x, None, None


class Avg(MessagePassing):
    def __init__(self, use_edge_attr):
        super(Avg, self).__init__()

        self.use_edge_attr = use_edge_attr

    def forward(self, data, proteins):
        y = data.y
        train_index = torch.nonzero(data.train_mask).squeeze(-1)

        edge_index, edge_weight = data.edge_index, data.edge_attr

        label = torch.zeros_like(y)
        label[train_index] = y[train_index]

        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=y.size(0), add_self_loops=False
        )

        if self.use_edge_attr:
            output = self.propagate(
                edge_index=edge_index, x=label, edge_weight=edge_weight, size=None
            )
        else:
            output = self.propagate(
                edge_index=edge_index, x=label, edge_weight=None, size=None
            )

        output[train_index] = y[train_index]

        return output, None, None


class MlpKnnGnn(nn.Module):

    def __init__(
            self, args, k : int, in_channels: int, out_channels: int = 1,
            mlp_layers: int = 2, mlp_hid_channels: int = -1, mlp_dropout: float = 0.5,
            gnn_layers: int = 2, gnn_hid_channels: int = -1, gnn_dropout: float = 0.5, beta: bool = True,
            flow: str = "source_to_target", loop: bool = True,
            force_undirected: bool = False,
    ):
        super(MlpKnnGnn, self).__init__()
        self.knn_k = k
        self.flow = flow
        self.loop = loop
        self.force_undirected = force_undirected

        mlp_hid_channels = in_channels if mlp_hid_channels < 0 else mlp_hid_channels

        self.mlp = MultiLayerPerceptron(
            in_channels=in_channels, hid_channels=mlp_hid_channels, out_channels=mlp_hid_channels,
            n_layers=mlp_layers, dropout=mlp_dropout
        )
        gnn_hid_channels = in_channels if gnn_hid_channels < 0 else gnn_hid_channels

        if args.gnn == "GCN":
            self.gnn = GCNEncoder(
                in_channels=mlp_hid_channels, hidden_channels=gnn_hid_channels, out_channels=out_channels,
                n_layers=gnn_layers, dropout=gnn_dropout
            )
        elif args.gnn == "GAT":
            self.gnn = GATEncoder(
                in_channels=mlp_hid_channels, hidden_channels=gnn_hid_channels, out_channels=out_channels,
                n_layers=gnn_layers, dropout=gnn_dropout, heads=8
            )
        else:
            raise ValueError("%s is not a valid gnn option." % args.gnn)

    def forward(self, data, proteins):

        features = data.x
        train_index = torch.nonzero(data.train_mask).squeeze(-1)
        valid_index = torch.nonzero(data.valid_mask).squeeze(-1)
        test_index = torch.nonzero(data.test_mask).squeeze(-1)

        x, _, _ = self.mlp(data=features, proteins=None)

        edge_index, edge_attr = knn_graph_fast(
            x=x, k=self.knn_k, loop=self.loop,
            flow=self.flow, force_undirected=self.force_undirected,
            cosine=False, only_from_train=False, no_test_test=False,
            train_index=train_index, valid_index=valid_index, test_index=test_index,
        )

        # x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.gnn(x=features, edge_index=edge_index, edge_weight=edge_attr)

        return x, edge_index, edge_attr


class MlpKnnGnnFeatRec(nn.Module):

    def __init__(
            self, args, k : int, in_channels: int, out_channels: int = 1,
            mlp_layers: int = 2, mlp_hid_channels: int = -1, mlp_dropout: float = 0.5,
            gnn_layers: int = 2, gnn_hid_channels: int = -1, gnn_dropout: float = 0.5,
            noise: str = "mask",
            flow: str = "source_to_target", loop: bool = True,
            force_undirected: bool = False,
    ):
        super(MlpKnnGnnFeatRec, self).__init__()
        self.knn_k = k
        self.noise = noise
        self.flow = flow
        self.loop = loop
        self.force_undirected = force_undirected

        mlp_hid_channels = in_channels if mlp_hid_channels < 0 else mlp_hid_channels

        self.mlp = MultiLayerPerceptron(
            in_channels=in_channels, hid_channels=mlp_hid_channels, out_channels=mlp_hid_channels,
            n_layers=mlp_layers, dropout=mlp_dropout
        )
        gnn_hid_channels = in_channels if gnn_hid_channels < 0 else gnn_hid_channels

        if args.gnn == "GCN":
            self.gnn_main = GCNEncoder(
                in_channels=mlp_hid_channels, hidden_channels=gnn_hid_channels, out_channels=out_channels,
                n_layers=gnn_layers, dropout=gnn_dropout
            )
        elif args.gnn == "GAT":
            self.gnn_main = GATEncoder(
                in_channels=mlp_hid_channels, hidden_channels=gnn_hid_channels, out_channels=out_channels,
                n_layers=gnn_layers, dropout=gnn_dropout, heads=8
            )
        else:
            raise ValueError("%s is not a valid gnn option." % args.gnn)

        if args.gnn == "GCN":
            self.gnn_rec = GCNEncoder(
                in_channels=mlp_hid_channels, hidden_channels=gnn_hid_channels, out_channels=in_channels,
                n_layers=gnn_layers, dropout=gnn_dropout
            )
        elif args.gnn == "GAT":
            self.gnn_rec = GATEncoder(
                in_channels=mlp_hid_channels, hidden_channels=gnn_hid_channels, out_channels=in_channels,
                n_layers=gnn_layers, dropout=gnn_dropout, heads=8
            )
        else:
            raise ValueError("%s is not a valid gnn option." % args.gnn)

    def forward(self, data, proteins, mask):

        features = data.x
        train_index = torch.nonzero(data.train_mask).squeeze(-1)
        valid_index = torch.nonzero(data.valid_mask).squeeze(-1)
        test_index = torch.nonzero(data.test_mask).squeeze(-1)

        if self.noise == "normal":
            noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
            masked_features = features + (noise * mask)
        elif self.noise == "mask":
            masked_features = features * (1 - mask)
        else:
            raise ValueError("Wrong noise choice")

        x, _, _ = self.mlp(data=features, proteins=None)

        edge_index, _ = knn_graph_fast(
            x=x, k=self.knn_k, loop=self.loop,
            flow=self.flow, force_undirected=self.force_undirected,
            cosine=False, only_from_train=False, no_test_test=False,
            train_index=train_index, valid_index=valid_index, test_index=test_index,
        )

        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=x.size(0), add_self_loops=False
        )

        x = self.gnn_main(x=features, edge_index=edge_index, edge_weight=edge_weight)
        x_rec = self.gnn_rec(x=masked_features, edge_index=edge_index, edge_weight=edge_weight)

        return x, x_rec, edge_index, edge_weight


class Transformer(nn.Module):

    def __init__(
            self, args, in_channels: int, out_channels: int = 1,
            gnn_layers: int = 2, gnn_hid_channels: int = -1, gnn_dropout: float = 0.5,
            heads: int = 8, beta: bool = True
    ):
        super(Transformer, self).__init__()
        self.use_edge_attr = args.use_edge_attr

        gnn_hid_channels = in_channels if gnn_hid_channels < 0 else gnn_hid_channels

        self.gnn = TransformerEncoder(
            in_channels=in_channels, hidden_channels=gnn_hid_channels, out_channels=out_channels,
            n_layers=gnn_layers, dropout=gnn_dropout, heads=heads, beta=beta
        )

    def forward(self, data, proteins):

        if self.use_edge_attr:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            x, edge_index, edge_attr = data.x, data.edge_index, None

        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)

        return x, None, None