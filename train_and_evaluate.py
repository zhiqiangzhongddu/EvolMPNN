import torch
from torch.nn import functional as F
from torch_geometric.data.data import Data
from torch_geometric.loader.neighbor_loader import NeighborLoader
import torchdrug.data as td_data

from utils import spearmanr, get_random_mask


def train_function(
        args, epoch, data, protein_feature, residue_feature, model, optimizer,
        n_id=None, sequences=None, device="cpu"
):

    assert type(data) == Data, "input data should be Data type"

    data.to(device)
    model.to(device)
    model.train()

    optimizer.zero_grad()

    if ("llm" in args.model.lower()) or ("cnn" in args.model.lower()) or ("bert" in args.model.lower()) or ("protbert" in args.model.lower()):
        proteins = td_data.PackedProtein.from_sequence(
            sequences, atom_feature=None, bond_feature=None, residue_feature="default"
        ).to(device)
    else:
        proteins = None

    if args.model == "MlpKnnGnnFeatRec":
        mask = get_random_mask(features=data.x, r=10, nr=5)
        indices = mask > 0

        output, out_rec, _edge_index, _edge_attr = model(data=data, proteins=proteins, mask=mask)
        mask = data["train_mask"]

        loss_main = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
        loss_rec = F.mse_loss(input=out_rec[indices], target=data.x[indices], reduction='mean')
        if epoch < args.adj_epochs:
            loss = loss_rec * args.lamda_rec
        else:
            loss = loss_main + loss_rec * args.lamda_rec

    elif args.model == "NodeFormer":

        if model.use_edge_loss:
            output, link_loss_ = model(x=data.x, adjs=data.adjs, tau=args.tau)
            mask = data["train_mask"]
            loss_main = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
            loss = loss_main - args.lamda_edge * sum(link_loss_) / len(link_loss_)
        else:
            output = model(x=data.x, adjs=data.adjs, tau=args.tau)
            mask = data["train_mask"]
            loss = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
        _edge_index = _edge_attr = None

    elif args.model == "ResidueMPNN":
        output = model(
            x_seq=protein_feature, x_residue=residue_feature, edge_index=data.edge_index,
            n_id=n_id
        )
        mask = data["train_mask"]
        loss = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
        _edge_index = _edge_attr = None

    elif args.model == "ResiduePMPNN":
        output = model(
            x_seq=protein_feature, x_residue=residue_feature,
            tau=args.tau, n_id=n_id
        )
        mask = data["train_mask"]
        loss = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
        _edge_index = _edge_attr = None

    elif args.model == "ResidueCnnPMPNN":
        output = model(
            proteins=proteins, x_residue=residue_feature,
            tau=args.tau, n_id=n_id
        )
        mask = data["train_mask"]
        loss = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
        _edge_index = _edge_attr = None

    elif args.model == "ResidueFormer":
        if model.use_edge_loss:
            output, link_loss_ = model(
                x_seq=protein_feature, x_residue=residue_feature, adjs=[None],
                tau=args.tau, n_id=n_id
            )
            mask = data["train_mask"]
            loss_main = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
            loss = loss_main - args.lamda_edge * sum(link_loss_) / len(link_loss_)
        else:
            output = model(
                x_seq=protein_feature, x_residue=residue_feature, adjs=[None],
                tau=args.tau, n_id=n_id
            )
            mask = data["train_mask"]
            loss = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
        _edge_index = _edge_attr = None

    elif args.model == "ResiduePFormer":
        if model.use_edge_loss:
            output, link_loss_ = model(
                x_seq=protein_feature, x_residue=residue_feature, adjs=[None],
                tau=args.tau, n_id=n_id
            )
            mask = data["train_mask"]
            loss_main = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
            loss = loss_main - args.lamda_edge * sum(link_loss_) / len(link_loss_)
        else:
            output = model(
                x_seq=protein_feature, x_residue=residue_feature, adjs=[None],
                tau=args.tau, n_id=n_id
            )
            mask = data["train_mask"]
            loss = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")
        _edge_index = _edge_attr = None

    else:
        output, _edge_index, _edge_attr = model(data=data, proteins=proteins)
        mask = data["train_mask"]
        loss = F.mse_loss(input=output[mask], target=data.y[mask], reduction="mean")

    loss.backward()
    optimizer.step()

    data.to('cpu')
    protein_feature.to('cpu') if protein_feature is not None else None
    residue_feature.to('cpu') if residue_feature is not None else None
    proteins.to('cpu') if proteins is not None else None
    del data, proteins, protein_feature, residue_feature
    torch.cuda.empty_cache()

    return loss.item(), _edge_index, _edge_attr


def train(args, epoch, data, protein_feature, residue_feature, model, optimizer, sequences=None, device="cpu"):

    if type(data) == Data:

        loss, _edge_index, _edge_attr = train_function(
            args=args, epoch=epoch, data=data, model=model,
            protein_feature=protein_feature, residue_feature=residue_feature,
            optimizer=optimizer, sequences=sequences, device=device
        )

    elif type(data) == NeighborLoader or type(data) == list:
        all_loss = 0

        for idx, batch in enumerate(data):
            if ("llm" in args.model.lower()) or ("cnn" in args.model.lower()) or ("bert" in args.model.lower()) or ("protbert" in args.model.lower()):
                batch_sequences = [sequences[idx] for idx in batch.n_id]
            else:
                batch_sequences = None
            batch['train_mask'][batch.batch_size:] = False

            if args.model in ["ResidueCnnPMPNN"]:
                batch_sequences += [sequences[idx] for idx in model.anchors]

            loss, _, _ = train_function(
                args=args, epoch=epoch, data=batch, model=model,
                protein_feature=protein_feature, residue_feature=residue_feature, n_id=batch.n_id,
                optimizer=optimizer, sequences=batch_sequences, device=device
            )
            all_loss += loss

        loss = all_loss / len(data)
        _edge_index = _edge_attr = None

    else:
        raise ValueError("Wrong data type")

    return loss, _edge_index, _edge_attr


@torch.no_grad()
def evaluate_function(args, data, protein_feature, residue_feature, model, n_id=None, sequences=None, device="cpu"):

    assert type(data) == Data, "input data should be Data type"

    if ("llm" in args.model.lower()) or ("cnn" in args.model.lower()) or ("bert" in args.model.lower()) or ("protbert" in args.model.lower()):
        proteins = td_data.PackedProtein.from_sequence(
            sequences, atom_feature=None, bond_feature=None, residue_feature="default"
        ).to(device)
    else:
        proteins = None

    data.to(device)
    model.to(device)
    model.eval()

    if args.model == "MlpKnnGnnFeatRec":
        pred, _, _, _ = model(data=data, proteins=proteins, mask=torch.zeros_like(data.x))

    elif args.model == "NodeFormer":
        if model.use_edge_loss:
            pred, _ = model(x=data.x, adjs=data.adjs, tau=args.tau)
        else:
            pred = model(x=data.x, adjs=data.adjs, tau=args.tau)

    elif args.model == "ResidueMPNN":
        pred = model(
            x_seq=protein_feature, x_residue=residue_feature, edge_index=data.edge_index,
            n_id=n_id
        )

    elif args.model == "ResiduePMPNN":
        pred = model(
            x_seq=protein_feature, x_residue=residue_feature,
            tau=args.tau, n_id=n_id
        )

    elif args.model == "ResidueCnnPMPNN":
        pred = model(
            proteins=proteins, x_residue=residue_feature,
            tau=args.tau, n_id=n_id
        )

    elif args.model == "ResidueFormer":
        if model.use_edge_loss:
            pred, _ = model(
                x_seq=protein_feature, x_residue=residue_feature, adjs=[None],
                tau=args.tau, n_id=n_id
            )
        else:
            pred = model(
                x_seq=protein_feature, x_residue=residue_feature, adjs=[None],
                tau=args.tau, n_id=n_id
            )

    elif args.model == "ResiduePFormer":
        if model.use_edge_loss:
            pred, _ = model(
                x_seq=protein_feature, x_residue=residue_feature, adjs=[None],
                tau=args.tau, n_id=n_id
            )
        else:
            pred = model(
                x_seq=protein_feature, x_residue=residue_feature, adjs=[None],
                tau=args.tau, n_id=n_id
            )

    else:
        pred, _, _ = model(data=data, proteins=proteins)

    data.to('cpu')
    protein_feature.to('cpu') if protein_feature is not None else None
    residue_feature.to('cpu') if residue_feature is not None else None
    proteins.to('cpu') if proteins is not None else None
    del data, proteins, protein_feature, residue_feature
    torch.cuda.empty_cache()

    return pred.cpu()


@torch.no_grad()
def evaluate(args, data, protein_feature, residue_feature, model, target_index=None, sequences=None, device="cpu"):

    if type(data) == Data:

        pred = evaluate_function(
            args=args, data=data, model=model,
            protein_feature=protein_feature, residue_feature=residue_feature,
            sequences=sequences, device=device,
        )
        score = spearmanr(pred=pred[target_index].squeeze(-1), target=data.y[target_index].squeeze(-1))

    elif type(data) == NeighborLoader or type(data) == list:

        preds = []
        targets = []

        for idx, batch in enumerate(data):
            if ("llm" in args.model.lower()) or ("cnn" in args.model.lower()) or ("bert" in args.model.lower()) or ("protbert" in args.model.lower()):
                batch_sequences = [sequences[idx] for idx in batch.n_id]
            else:
                batch_sequences = None

            _pred = evaluate_function(
                args=args, data=batch, model=model,
                protein_feature=protein_feature, residue_feature=residue_feature, n_id=batch.n_id,
                sequences=batch_sequences, device=device,
            )
            preds.append(_pred.squeeze(-1)[:batch.batch_size])
            targets.append(batch.y.squeeze(-1)[:batch.batch_size])

        score = spearmanr(pred=torch.cat(preds, dim=-1), target=torch.cat(targets, dim=-1))

    else:
        raise ValueError("Wrong data type")

    return score
