#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm

import torch
from torch_geometric.loader import NeighborLoader
import torchdrug.data as td_data

from data import load_data
from build_pipeline import build_dataloader, build_model
from train_and_evaluate import evaluate
from utils import parse_args, set_seed, make_output_folder, get_root_logger

import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# In[1]:





# In[2]:


args = parse_args()
set_seed(args.seed)

# args.gpu_id = 1
# args.dataset = "AAV" # GB1, Fluorescence, AAV
args.batch_size = 32
# args.split = "one_vs_rest"
args.oh_residue_feat = False
args.full_residue_feat = False
# args.model = "ProtBertMlp"
# args.encoder = "ESM-1b"
args.pretrained_encoder = True
args.pretrained_mlp = True

args.device = torch.device(
    'cuda:%s' % args.gpu_id if torch.cuda.is_available() else 'cpu'
)
output_folder = make_output_folder(args=args)
logger = get_root_logger(folder=output_folder)
logger.warning(args)


# In[2]:





# In[3]:


data, _, _, sequences, _, _ = load_data(
    args=args, logger=logger, 
    feature_generator=args.feature_generator, pretrained=args.fine_tuned_generator, 
    oh_residue_feat=False, full_residue_feat=False, load_protein_feat=False
)


# In[4]:


model = build_model(
    args=args, in_channels=data.num_features, residue_dim=data.residue_dim, out_channels=data.y.size(1), logger=logger,
).to(args.device)
logger.warning(model)


# In[ ]:





# In[ ]:


train_loader, valid_loader, test_loader = build_dataloader(
    data=data, batch_size=args.batch_size, num_neighbors=[0], shuffle=False
)

train_score = evaluate(
    args=args, data=train_loader, model=model, 
    protein_feature=None, residue_feature=None,
    sequences=sequences, device=args.device
)
valid_score = evaluate(
    args=args, data=valid_loader, model=model, 
    protein_feature=None, residue_feature=None,
    sequences=sequences, device=args.device
)
test_score = evaluate(
    args=args, data=test_loader, model=model, 
    protein_feature=None, residue_feature=None,
    sequences=sequences, device=args.device
)

logger.warning("Train {:.5f}, valid {:.5f}, test {:.5f}".format(
train_score, valid_score, test_score
))


# In[ ]:





# In[ ]:


def get_features(model, data, sequences):
    model.eval()

    protein_feature = []

    for batch in tqdm(data):
        batch_sequences = [sequences[idx] for idx in batch.n_id]
        batch_proteins = td_data.PackedProtein.from_sequence(
            batch_sequences, atom_feature=None, bond_feature=None, residue_feature="default"
        ).to(args.device)

        output = model.encoder(graph=batch_proteins, input=batch_proteins.node_feature.float(), all_loss=None, metric=None)["graph_feature"].detach().cpu()
        protein_feature.append(output)

    return torch.cat(protein_feature, dim=0)


# In[ ]:





# In[ ]:


data_loader = NeighborLoader(
    data=data, input_nodes=torch.tensor([True] * data.num_nodes),
    num_neighbors=[0], batch_size=args.batch_size, shuffle=False,
    # num_workers=12, persistent_workers=True,
)
data_loader = [batch_data for batch_data in data_loader]
protein_feature = get_features(model=model, data=data_loader, sequences=sequences)


# In[ ]:





# In[ ]:


feature_folder = "/home/zhiqiang/Homology-PyG/input/protein_features"
if args.pretrained_encoder:
    protein_file_path = os.path.join(
        feature_folder, "protein_features_%s_%s_%s.pt" % (args.dataset, args.encoder, args.split)
    )
else:
    protein_file_path = os.path.join(
        feature_folder, "protein_features_%s_%s.pt" % (args.dataset, args.encoder)
    )
print('saving protein feature at %s' % protein_file_path)
torch.save(protein_feature, protein_file_path)

# if args.pretrained_encoder:
#     residue_file_path = os.path.join(
#         feature_folder, "residue_features_%s_%s_%s.pt" % (args.dataset, args.encoder, args.split)
#     )
# else:
#     residue_file_path = os.path.join(
#         feature_folder, "residue_features_%s_%s.pt" % (args.dataset, args.model)
#     )
# print('saving residue feature at %s' % residue_file_path)
# torch.save(residue_feature, residue_file_path)


# In[ ]:




