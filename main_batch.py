import os
import torch
import torch_geometric as tg
from torch_geometric.utils import remove_self_loops, add_self_loops

from data import load_data
from graph_generator import HomologyGraphStats
from build_pipeline import build_dataloader, build_model, build_optimizer
from train_and_evaluate import train, evaluate
from utils import parse_args, set_seed, check_adjmatrix, make_output_folder, get_root_logger, adj_mul, save_info

import warnings
warnings.filterwarnings('ignore')
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

args = parse_args()
set_seed(args.seed)

args.device = torch.device(
    'cuda:%s' % args.gpu_id if torch.cuda.is_available() else 'cpu'
)
output_folder = make_output_folder(args=args)
logger = get_root_logger(folder=output_folder)
logger.warning(args)


# set up dataset
data, protein_feature, residue_feature, sequences, wt_sequence, similarity_matrix = load_data(
    args=args, logger=logger, load_protein_feat=args.load_protein_feat,
    feature_generator=args.feature_generator, pretrained=args.fine_tuned_generator,
    oh_residue_feat=args.oh_residue_feat, full_residue_feat=args.full_residue_feat
)
args.wt_index = sequences.index(wt_sequence)

# print("is saving feat file")
# torch.save(residue_feature, "../input/residue_features_%s-%s.pt" % (args.dataset, args.split))

if args.model in ["Gnn", "Avg", "GnnMlp", "Transformer", "NodeFormer",
                  "ResidueMPNN", "ResidueFormer"]:
    logger.warning("Initialising graph structure.")
    homology_stats_graph = HomologyGraphStats(
        dataset=args.dataset, homology=args.homology, knn_k=args.knn_k, loop=args.loop, flow=args.flow,
        force_undirected=args.force_undirected,
        only_from_train=args.only_from_train, no_test_test=args.no_test_test,
        train_mask=data.train_mask, valid_mask=data.valid_mask, test_mask=data.test_mask
    )
    data.edge_index, data.edge_attr = homology_stats_graph(x=data.x, logger=logger, similarity_matrix=similarity_matrix)
    logger.warning("Average in-degree: %s" % tg.utils.degree(data.edge_index[1]).mean().item())
    logger.warning(check_adjmatrix(data))

if args.batch_size > 0:
    train_loader, valid_loader, test_loader = build_dataloader(
        data=data, batch_size=args.batch_size, num_neighbors=[-1], shuffle=True
    )
    if args.model in ["NodeFormer", "ResidueFormer"]:
        for loader in [train_loader, valid_loader, test_loader]:
            for idx, batch in enumerate(loader):
                ### Adj storage for relational bias ###
                adjs = []
                adj, _ = remove_self_loops(batch.edge_index)
                adj, _ = add_self_loops(adj, num_nodes=batch.num_nodes)
                adjs.append(adj)
                for i in range(args.rb_order - 1): # edge_index of high order adjacency
                    adj = adj_mul(adj, adj, batch.num_nodes)
                    adjs.append(adj)
                batch.adjs = adjs
                loader[idx] = batch
else:
    train_loader, valid_loader, test_loader = None, None, None
    if args.model in ["NodeFormer", "ResidueFormer"]:
        ### Adj storage for relational bias ###
        adjs = []
        adj, _ = remove_self_loops(data.edge_index)
        adj, _ = add_self_loops(adj, num_nodes=data.num_nodes)
        adjs.append(adj)
        for i in range(args.rb_order - 1):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, data.num_nodes)
            adjs.append(adj)
        data.adjs = adjs

if args.model in ["ResidueMPNN", "ResiduePMPNN", "ResidueCnnPMPNN", "ResidueFormer", "ResiduePFormer"]:
    args.num_residue = max([len(seq) for seq in sequences])
    args.num_sequences = len(sequences)
logger.warning(data)


# set up model
model = build_model(
    args=args, in_channels=data.num_features, residue_dim=data.residue_dim, out_channels=data.y.size(1), logger=logger,
).to(args.device)
logger.warning(model)
optimizer = build_optimizer(args=args, model=model, logger=logger)

best_epoch = best_train = best_val = best_test = 0
for epoch in range(1, args.epochs + 1):
    if args.batch_size > 0:
        loss, _, _ = train(
            args=args, epoch=epoch, data=train_loader, model=model,
            protein_feature=protein_feature, residue_feature=residue_feature,
            optimizer=optimizer, sequences=sequences, device=args.device
        )
    else:
        loss, _, _ = train(
            args=args, epoch=epoch, data=data, model=model,
            protein_feature=protein_feature, residue_feature=residue_feature,
            optimizer=optimizer, sequences=sequences, device=args.device
        )
    logger.warning("epoch {}, loss {:.5f}".format(epoch, loss))

    if args.batch_size > 0:
        train_score = evaluate(
            args=args, data=train_loader, model=model,
            protein_feature=protein_feature, residue_feature=residue_feature,
            sequences=sequences, device=args.device
        )
        valid_score = evaluate(
            args=args, data=valid_loader, model=model,
            protein_feature=protein_feature, residue_feature=residue_feature,
            sequences=sequences, device=args.device
        )
    else:
        train_index = torch.nonzero(data.train_mask).squeeze()
        train_score = evaluate(
            args=args, data=data, model=model,
            protein_feature=protein_feature, residue_feature=residue_feature,
            target_index=train_index, sequences=sequences, device=args.device
        )
        valid_index = torch.nonzero(data.valid_mask).squeeze()
        valid_score = evaluate(
            args=args, data=data, model=model,
            protein_feature=protein_feature, residue_feature=residue_feature,
            target_index=valid_index, sequences=sequences, device=args.device
        )

    if valid_score > best_val:
        logger.warning("updating best performance.")
        if args.batch_size > 0:
            test_score = evaluate(
                args=args, data=test_loader, model=model,
                protein_feature=protein_feature, residue_feature=residue_feature,
                sequences=sequences, device=args.device
            )
        else:
            test_index = torch.nonzero(data.test_mask).squeeze()
            test_score = evaluate(
                args=args, data=data, model=model,
                protein_feature=protein_feature, residue_feature=residue_feature,
                target_index=test_index, sequences=sequences, device=args.device
            )
        best_epoch, best_train, best_val, best_test = epoch, train_score, valid_score, test_score
        logger.warning("epoch {}, train {:.5f}, valid {:.5f}, test {:.5f}".format(
            best_epoch, train_score, best_val, best_test
        ))
    else:
        logger.warning("epoch {}, train {:.5f}, valid {:.5f}".format(
            epoch, train_score, valid_score
        ))

logger.warning("Best epoch {}, train {:.5f}, valid {:.5f}, test {:.5f}".format(
    best_epoch, best_train, best_val, best_test
))
save_info(
    args=args, folder=output_folder, result=best_test.item()
)
