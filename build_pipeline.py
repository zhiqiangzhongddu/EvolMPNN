import os

import torch
from torch_geometric.loader import NeighborLoader

from model import (
    Avg, Gnn, LlmGnnMlp, LlmMlp, CnnMlp, BertMlp, ProtBertMlp,
    GnnMlp, MultiLayerPerceptron, MlpKnnGnn, MlpKnnGnnFeatRec, Transformer
)
from nodeformer import NodeFormer
from residuempnn import ResidueMPNN
from residueformer import ResidueFormer
from residuepmpnn import ResiduePMPNN

model_parameters_folder = "./input/model_parameters"


def build_model(args, in_channels, out_channels, residue_dim, logger):
    if args.model == "Avg":
        model = Avg(
            use_edge_attr=args.use_edge_attr
        )
    elif args.model == "Mlp":
        model = MultiLayerPerceptron(
            in_channels=in_channels, hid_channels=args.mlp_hid_channels, out_channels=out_channels,
            n_layers=args.mlp_layers, dropout=args.mlp_dropout
        )
    elif args.model == "Gnn":
        model = Gnn(
            args=args, in_channels=in_channels, out_channels=out_channels,
            gnn_layers=args.gnn_layers, gnn_hid_channels=args.gnn_hid_channels, gnn_dropout=args.gnn_dropout,
        )
    elif args.model == "GnnMlp":
        model = GnnMlp(
            args=args, in_channels=in_channels, out_channels=out_channels,
            gnn_layers=args.gnn_layers, gnn_hid_channels=args.gnn_hid_channels, gnn_dropout=args.gnn_dropout,
            mlp_layers=args.mlp_layers, mlp_hid_channels=args.mlp_hid_channels, mlp_dropout=args.mlp_dropout
        )
    elif args.model == "LlmMlp":
        model = LlmMlp(
            args=args, llm_model=args.encoder, out_channels=out_channels,
            mlp_layers=args.mlp_layers, mlp_dropout=args.mlp_dropout
        )
    elif args.model == "LlmGnnMlp":
        model = LlmGnnMlp(
            args=args, llm_model=args.encoder, out_channels=out_channels,
            gnn_layers = args.gnn_layers, gnn_hid_channels=args.gnn_hid_channels, gnn_dropout=args.gnn_dropout,
            mlp_layers=args.mlp_layers, mlp_dropout=args.mlp_dropout,
        )
    elif args.model == "CnnMlp":
        model = CnnMlp(
            args=args,
            mlp_layers=args.mlp_layers, mlp_dropout=args.mlp_dropout
        )
    elif args.model == "BertMlp":
        model = BertMlp(
            args=args,
            mlp_layers=args.mlp_layers, mlp_dropout=args.mlp_dropout
        )
    elif args.model == "ProtBertMlp":
        model = ProtBertMlp(
            args=args,
            mlp_layers=args.mlp_layers, mlp_dropout=args.mlp_dropout
        )
    elif args.model == "MlpKnnGnn":
        model = MlpKnnGnn(
            args=args, k=args.knn_k, in_channels=in_channels, out_channels=out_channels,
            mlp_layers=args.mlp_layers, mlp_hid_channels=args.mlp_hid_channels, mlp_dropout=args.mlp_dropout,
            gnn_layers = args.gnn_layers, gnn_hid_channels = args.gnn_hid_channels, gnn_dropout = args.gnn_dropout,
        )
    elif args.model == "MlpKnnGnnFeatRec":
        model = MlpKnnGnnFeatRec(
            args=args, k=args.knn_k, in_channels=in_channels, out_channels=out_channels,
            mlp_layers=args.mlp_layers, mlp_hid_channels=args.mlp_hid_channels, mlp_dropout=args.mlp_dropout,
            gnn_layers = args.gnn_layers, gnn_hid_channels = args.gnn_hid_channels, gnn_dropout = args.gnn_dropout,
        )
    elif args.model == "Transformer":
        model = Transformer(
            args=args, in_channels=in_channels, out_channels=out_channels,
            gnn_layers=args.gnn_layers, gnn_hid_channels=args.gnn_hid_channels, gnn_dropout=args.gnn_dropout,
        )
    elif args.model == "NodeFormer":
        model = NodeFormer(
            in_channels=in_channels, out_channels=out_channels,
            num_layers=args.gnn_layers, hidden_channels=args.gnn_hid_channels,
            num_heads=args.num_heads, dropout=args.gnn_dropout,
            nb_random_features=30, use_bn=args.use_bn,
            use_residual=args.use_residual, use_act=True, use_jk=args.use_jk,
            use_gumbel=True, nb_gumbel_sample=10,
            rb_order=args.rb_order, rb_trans='sigmoid', use_edge_loss=args.use_edge_loss
        )
    elif args.model == "EvolGNN":
        model = ResidueMPNN(
            in_channels=in_channels, out_channels=out_channels, residue_dim=residue_dim,
            num_gnn_layers=args.gnn_layers, num_former_layers=args.former_layers, hidden_channels=args.gnn_hid_channels,
            num_residue=args.num_residue, num_sequences=args.num_sequences,
            num_heads=args.num_heads, dropout=args.gnn_dropout, use_bn=args.use_bn,
            use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
        )
    elif args.model == "EvolMPNN":
        model = ResiduePMPNN(
            in_channels=in_channels, out_channels=out_channels, residue_dim=residue_dim,
            num_gnn_layers=args.gnn_layers, num_former_layers=args.former_layers, hidden_channels=args.gnn_hid_channels,
            num_residue=args.num_residue, num_sequences=args.num_sequences, num_anchor=args.num_anchor,
            dropout=args.gnn_dropout, use_bn=args.use_bn,
            use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
        )
    elif args.model == "EvolFormer":
        model = ResidueFormer(
            in_channels=in_channels, out_channels=out_channels, residue_dim=residue_dim,
            num_layers=args.gnn_layers, hidden_channels=args.gnn_hid_channels,
            num_residue=args.num_residue, num_sequences=args.num_sequences,
            num_heads=args.num_heads, dropout=args.gnn_dropout,
            nb_random_features=30, use_bn=args.use_bn,
            use_residual=args.use_residual,
            use_act=args.use_act, use_jk=args.use_jk,
            use_gumbel=True, nb_gumbel_sample=10,
            rb_order=args.rb_order, rb_trans='sigmoid', use_edge_loss=args.use_edge_loss,
            num_anchor=args.num_anchor,
        )
    else:
        raise ValueError("%s is not a valid model" % args.model)

    if "Llm" in args.model and args.pretrained_encoder:
        file_path = os.path.join(model_parameters_folder,
                                 "%s-%s/Llm_Mlp_%s.pth" % (args.dataset, args.encoder, args.split))
        logger.warning("loading model parameters at %s" % file_path)
        state = torch.load(file_path, map_location="cpu")

        if "Llm" in args.model and args.pretrained_encoder:
            if state['Llm_model']['mapping'].size != model.encoder.state_dict()['mapping'].size:
                state['Llm_model']['mapping'] = model.encoder.state_dict()['mapping']
            model.encoder.load_state_dict(state["Llm_model"])
            logger.warning("LLM model parameters loaded")

        if "Mlp" in args.model and args.pretrained_mlp:
            model.mlp.load_state_dict(state["Mlp_model"])
            logger.warning("MLP model parameters loaded")

    if "Cnn" in args.model and args.pretrained_encoder:
        file_path = os.path.join(model_parameters_folder,
                                 "%s-CNN/Cnn_Mlp_%s.pth" % (args.dataset, args.split))
        logger.warning("loading model parameters at %s" % file_path)
        state = torch.load(file_path, map_location="cpu")

        if "Cnn" in args.model and args.pretrained_encoder:
            model.encoder.load_state_dict(state["Llm_model"])
            logger.warning("CNN model parameters loaded")

        if "Mlp" in args.model and args.pretrained_mlp:
            model.mlp.load_state_dict(state["Mlp_model"])
            logger.warning("MLP model parameters loaded")

    if "Bert" in args.model and "ProtBert" not in args.model and args.pretrained_encoder:
        file_path = os.path.join(model_parameters_folder,
                                 "%s-BERT/Llm_Mlp_%s.pth" % (args.dataset, args.split))
        logger.warning("loading model parameters at %s" % file_path)
        state = torch.load(file_path, map_location="cpu")

        if "Bert" in args.model and args.pretrained_encoder:
            model.encoder.load_state_dict(state["Llm_model"])
            logger.warning("BERT model parameters loaded")

        if "Mlp" in args.model and args.pretrained_mlp:
            model.mlp.load_state_dict(state["Mlp_model"])
            logger.warning("MLP model parameters loaded")

    if "ProtBert" in args.model and args.pretrained_encoder:
        file_path = os.path.join(model_parameters_folder,
                                 "%s-ProtBert/Llm_Mlp_%s.pth" % (args.dataset, args.split))
        logger.warning("loading model parameters at %s" % file_path)
        state = torch.load(file_path, map_location="cpu")

        if "ProtBert" in args.model and args.pretrained_encoder:
            model.encoder.load_state_dict(state["Llm_model"])
            logger.warning("ProtBert model parameters loaded")

        if "Mlp" in args.model and args.pretrained_mlp:
            model.mlp.load_state_dict(state["Mlp_model"])
            logger.warning("MLP model parameters loaded")

    if ("Llm" in args.model or "Cnn" in args.model or "Bert" in args.model or "ProtBert" in args.model) and args.fix_encoder:
        logger.warning("ENCODER model params are frozen")
        for p in model.encoder.parameters():
            p.requires_grad = False

    if "Mlp" == args.model and args.fix_mlp:
        logger.warning("MLP model params are frozen")
        for p in model.parameters():
            p.requires_grad = False

    if "Mlp" != args.model and "Mlp" in args.model and args.fix_mlp:
        logger.warning("MLP model params are frozen")
        for p in model.mlp.parameters():
            p.requires_grad = False

    return model


def build_optimizer(args, model, logger):

    if args.optimizer == "RMSProp":
        if args.model in ["LlmMlp"] and args.lr_ratio:
            optimizer = torch.optim.RMSprop(
                params=[
                    {'params': model.encoder.parameters(), 'lr': args.lr * args.lr_ratio},
                    {'params': model.mlp.parameters(), 'lr': args.lr}
                ], lr=args.lr, eps=1e-08
            )
        elif args.model in ["LlmGnnMlp"] and args.lr_ratio:
            optimizer = torch.optim.RMSprop(
                params=[
                    {'params': model.encoder.parameters(), 'lr': args.lr * args.lr_ratio},
                    {'params': model.gnn.parameters(), 'lr': args.lr},
                    {'params': model.mlp.parameters(), 'lr': args.lr}
                ], lr=args.lr, eps=1e-08
            )
        else:
            optimizer = torch.optim.RMSprop(
                params=model.parameters(), lr=args.lr, eps=1e-08
            )
    elif args.optimizer == "Adam":
        if args.model in ["LlmMlp"] and args.lr_ratio:
            optimizer = torch.optim.Adam(
                params=[
                    {'params': model.encoder.parameters(), 'lr': args.lr * args.lr_ratio},
                    {'params': model.mlp.parameters(), 'lr': args.lr}
                ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
            )
        elif args.model in ["LlmGnnMlp"] and args.lr_ratio:
            optimizer = torch.optim.Adam(
                params=[
                    {'params': model.encoder.parameters(), 'lr': args.lr * args.lr_ratio},
                    {'params': model.gnn.parameters(), 'lr': args.lr},
                    {'params': model.mlp.parameters(), 'lr': args.lr}
                ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
            )
        else:
            optimizer = torch.optim.Adam(
                params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
    else:
        raise ValueError("%s is not a valid optimizer" % args.optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer=optimizer, gamma=args.decay_rate
    # )

    return optimizer


def build_dataloader(data, batch_size, num_neighbors=[-1], shuffle=False):

    train_loader = NeighborLoader(
        data=data, input_nodes=data.train_mask,
        num_neighbors=num_neighbors, batch_size=batch_size, shuffle=shuffle,
        # num_workers=12, persistent_workers=True,
    )
    train_loader = [batch_data for batch_data in train_loader]

    valid_loader = NeighborLoader(
        data=data, input_nodes=data.valid_mask,
        num_neighbors=num_neighbors, batch_size=batch_size, shuffle=shuffle,
        # num_workers=12, persistent_workers=True,
    )
    valid_loader = [batch_data for batch_data in valid_loader]

    test_loader = NeighborLoader(
        data=data, input_nodes=data.test_mask,
        num_neighbors=num_neighbors, batch_size=batch_size, shuffle=shuffle,
        # num_workers=12, persistent_workers=True,
    )
    test_loader = [batch_data for batch_data in test_loader]

    return train_loader, valid_loader, test_loader
