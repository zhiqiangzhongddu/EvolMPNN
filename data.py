import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

import torch
from torch_geometric.data import Data
import torchdrug.models as tg_model
import torchdrug.data as td_data

from utils import slice_indices, onehot


data_folder = "./data"
similarity_folder = "./input"
model_param_folder = "./input/model_parameters"
esm_folder = "./input/esm-model-weights"
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
residue_symbol2id = {"G": 0, "A": 1, "S": 2, "P": 3, "V": 4, "T": 5, "C": 6, "I": 7, "L": 8, "N": 9,
            "D": 10, "Q": 11, "K": 12, "E": 13, "M": 14, "H": 15, "F": 16, "R": 17, "Y": 18, "W": 19}
output_dim = {
    "ESM-1b": 1280,
    "ESM-1v": 1280,
    "ESM-2-8M": 320,
    "ESM-2-35M": 480,
    "ESM-2-150M": 640,
    "ESM-2-650M": 1280,
    "ESM-2-3B": 2560,
    "ESM-2-15B": 5120,
}


def load_data(
        args, logger,
        feature_generator, pretrained, oh_residue_feat, full_residue_feat,
        load_protein_feat
):
    if args.dataset.lower() == "gb1":
        data, protein_feature, residue_feature, sequences, reference_seq = load_gb1(
            args=args, logger=logger, load_protein_feat=load_protein_feat,
            feature_generator=feature_generator,
            pretrained=pretrained,
            oh_residue_feat=oh_residue_feat, full_residue_feat=full_residue_feat
        )
        if data.x is not None:
            assert data.x.size(0) == data.y.size(0) == 8733

    elif args.dataset.lower() == "aav":
        data, protein_feature, residue_feature, sequences, reference_seq = load_aav(
            args=args, logger=logger, load_protein_feat=load_protein_feat,
            feature_generator=feature_generator,
            pretrained=pretrained,
            oh_residue_feat=oh_residue_feat, full_residue_feat=full_residue_feat
        )
        if data.x is not None:
            assert data.x.size(0) == data.y.size(0) == 82583

    elif args.dataset.lower() == "fluorescence":
        data, protein_feature, residue_feature, sequences, reference_seq = load_fluorescence(
            args=args, logger=logger, load_protein_feat=load_protein_feat,
            feature_generator=feature_generator,
            pretrained=pretrained,
            oh_residue_feat=oh_residue_feat, full_residue_feat=full_residue_feat
        )
        if data.x is not None:
            assert data.x.size(0) == data.y.size(0) == 54025
    else:
        raise ValueError("%s is not a valid dataset" % args.dataset)

    data.num_nodes = data.y.size(0)
    data.residue_dim = residue_feature.size(-1) if residue_feature is not None else 0

    if args.homology == "gzip":
        similarity_matrix = load_similarity_matrix(
            similarity=args.homology, dataset=args.dataset.lower(),
            logger=logger
        )
    else:
        similarity_matrix = None

    return data, protein_feature, residue_feature, sequences, reference_seq, similarity_matrix


def load_gb1(
        args, logger,
        feature_generator, pretrained, oh_residue_feat, full_residue_feat,
        load_protein_feat=True
):
    assert args.split in ['one_vs_rest', 'two_vs_rest', 'three_vs_rest', 'low_vs_high',
                          'sampled'], "%s is not a valid split" % args.split

    # gb1_ref_seq_file = data_folder + "/FLIP/gb1/5LDE_1.fasta"
    if args.split == "one_vs_rest":
        file_path = data_folder + "/FLIP/gb1/splits/one_vs_rest.csv"
    elif args.split == "two_vs_rest":
        file_path = data_folder + "/FLIP/gb1/splits/two_vs_rest.csv"
    elif args.split == "three_vs_rest":
        file_path = data_folder + "/FLIP/gb1/splits/three_vs_rest.csv"
    elif args.split == "low_vs_high":
        file_path = data_folder + "/FLIP/gb1/splits/low_vs_high.csv"
    elif args.split("sampled"):
        file_path = data_folder + "/FLIP/gb1/splits/sampled.csv"
    else:
        raise ValueError("%s is not a valid split" % args.split)

    # gb1_region = (2, 56)
    gb1_reference_seq = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTELEVLFQGPLDPNSM" \
                        "ATYEVLCEVARKLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRD" \
                        "LLHLDPRFLERHLAGTMSYFSPYQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLD" \
                        "LLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLHLSGPSHSQHYRHTPLEHHHHHH"

    df_split_data = pd.read_csv(file_path)
    df_train = df_split_data[(df_split_data["set"] == "train") & (df_split_data["validation"] != True)]
    df_valid = df_split_data[df_split_data["validation"] == True]
    df_test = df_split_data[df_split_data["set"] == "test"]

    sequences = df_split_data["sequence"].values.tolist()

    train_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    train_mask[df_train.index.values] = True
    valid_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    valid_mask[df_valid.index.values] = True
    test_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    test_mask[df_test.index.values] = True

    protein_feature, residue_feature = load_protein_residue_feature(
        args=args, sequences=sequences, logger=logger,
        feature_generator=feature_generator, pretrained=pretrained,
        oh_residue_feat=oh_residue_feat, full_residue_feat=full_residue_feat,
        load_protein_feat=load_protein_feat
    )

    data = Data(
        y=torch.FloatTensor(df_split_data["target"].values).unsqueeze(-1),
        x=protein_feature,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
    )

    return data, protein_feature, residue_feature, sequences, gb1_reference_seq


def load_aav(
        args, logger,
        feature_generator, pretrained, oh_residue_feat, full_residue_feat,
        keep_mutation_region=True, load_protein_feat=True
):

    # assert args.split in ['des_mut', 'low_vs_high', 'mut_des', 'one_vs_rest', 'sampled', 'seven_vs_rest', 'two_vs_rest']
    assert args.split in ['low_vs_high', 'one_vs_rest', 'sampled', 'seven_vs_rest',
                          'two_vs_rest'], "%s is not a valid split" % args.split

    aav_ref_seq_file = data_folder + "/FLIP/aav/P03135.fasta"
    if args.split == "one_vs_rest":
        file_path = data_folder + "/FLIP/aav/splits/one_vs_many.csv"
    elif args.split == "two_vs_rest":
        file_path = data_folder + "/FLIP/aav/splits/two_vs_many.csv"
    elif args.split == "seven_vs_rest":
        file_path = data_folder + "/FLIP/aav/splits/seven_vs_many.csv"
    elif args.split == "low_vs_high":
        file_path = data_folder + "/FLIP/aav/splits/low_vs_high.csv"
    elif args.split("sampled"):
        file_path = data_folder + "/FLIP/aav/splits/sampled.csv"
    else:
        raise ValueError("%s is not a valid split" % args.split)

    region = slice(474, 674)
    aav_reference_seq = str(next(SeqIO.parse(aav_ref_seq_file, "fasta")).seq)
    aav_reference_seq = aav_reference_seq[region]

    df_split_data = pd.read_csv(file_path)
    # remove sequences out of Train-Valid-Test
    df_split_data = df_split_data[df_split_data["set"].isin(['train', 'test'])].reset_index(drop=True)
    df_train = df_split_data[(df_split_data["set"] == "train") & (df_split_data["validation"] != True)]
    df_valid = df_split_data[df_split_data["validation"] == True]
    df_test = df_split_data[df_split_data["set"] == "test"]

    if keep_mutation_region:
        df_split_data["sequence"] = df_split_data["sequence"].apply(lambda seq: seq[region])
    sequences = df_split_data["sequence"].values.tolist()

    train_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    train_mask[df_train.index.values] = True
    valid_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    valid_mask[df_valid.index.values] = True
    test_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    test_mask[df_test.index.values] = True

    protein_feature, residue_feature = load_protein_residue_feature(
        args=args, sequences=sequences, logger=logger,
        feature_generator=feature_generator, pretrained=pretrained,
        oh_residue_feat=oh_residue_feat, full_residue_feat=full_residue_feat,
        load_protein_feat=load_protein_feat
    )

    data = Data(
        y=torch.FloatTensor(df_split_data["target"].values).unsqueeze(-1),
        x=protein_feature,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
    )

    return data, protein_feature, residue_feature, sequences, aav_reference_seq


def load_fluorescence(
        args, logger,
        feature_generator, pretrained, oh_residue_feat, full_residue_feat,
        load_protein_feat=True
):
    assert args.split in ['one_vs_rest', 'two_vs_rest', 'three_vs_rest', 'low_vs_high',
                          'sampled'], "%s is not a valid split" % args.split

    if args.split == "one_vs_rest":
        file_path = data_folder + "/PEER/fluorescence/splits/one_vs_many.csv"
    elif args.split == "two_vs_rest":
        file_path = data_folder + "/PEER/fluorescence/splits/two_vs_many.csv"
    elif args.split == "three_vs_rest":
        file_path = data_folder + "/PEER/fluorescence/splits/three_vs_many.csv"
    elif args.split == "low_vs_high":
        file_path = data_folder + "/PEER/fluorescence/splits/low_vs_high.csv"
    elif args.split("sampled"):
        file_path = data_folder + "/PEER/fluorescence/splits/sampled.csv"
    else:
        raise ValueError("%s is not a valid split" % args.split)

    fluorescence_reference_seq = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQ" \
                                 "CFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNIL" \
                                 "GHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQS" \
                                 "ALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

    df_split_data = pd.read_csv(file_path)
    df_train = df_split_data[(df_split_data["set"] == "train") & (df_split_data["validation"] != True)]
    df_valid = df_split_data[df_split_data["validation"] == True]
    df_test = df_split_data[df_split_data["set"] == "test"]

    sequences = df_split_data["primary"].values.tolist()

    # pad it with G while sequence is short
    max_len = max([len(seq) for seq in sequences])
    for idx, seq in enumerate(sequences):
        if len(seq) < max_len:
            while len(seq) < max_len:
                seq += 'G'
            sequences[idx] = seq

    train_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    train_mask[df_train.index.values] = True
    valid_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    valid_mask[df_valid.index.values] = True
    test_mask = torch.zeros(df_split_data.shape[0], dtype=torch.bool)
    test_mask[df_test.index.values] = True

    protein_feature, residue_feature = load_protein_residue_feature(
        args=args, sequences=sequences, logger=logger,
        feature_generator=feature_generator, pretrained=pretrained,
        oh_residue_feat=oh_residue_feat, full_residue_feat=full_residue_feat,
        load_protein_feat=load_protein_feat,
    )

    data = Data(
        y=torch.FloatTensor(df_split_data["log_fluorescence"].values).unsqueeze(-1),
        x=protein_feature,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
    )

    return data, protein_feature, residue_feature, sequences, fluorescence_reference_seq


@torch.no_grad()
def load_protein_residue_feature(
        args, sequences, logger, feature_generator, pretrained,
        oh_residue_feat, full_residue_feat,
        load_protein_feat = True
):
    # load protein features
    logger.warning("processing protein features")
    if load_protein_feat:
        if pretrained:
            protein_file_path = os.path.join(
                feature_folder, "protein_features_%s_%s_%s.pt" % (args.dataset, feature_generator, args.split)
            )
        else:
            protein_file_path = os.path.join(
                feature_folder, "protein_features_%s_%s.pt" % (args.dataset, feature_generator)
            )
        protein_feature = torch.load(f=protein_file_path)
        logger.warning("Loaded protein feature at %s" % protein_file_path)
    else:
        logger.warning("No protein feature")
        protein_feature = None

    # load residue features
    logger.warning("processing residue features")
    # use one-hot residue feature
    if oh_residue_feat:
        # max_len = max([len(seq) for seq in sequences])
        _oh_residue_feature = []
        sequences_tqdm = tqdm(sequences, "Initializing residue OneHot feature")
        for seq in sequences_tqdm:  # encode every sequence - N
            _oh_residue = [onehot(x=residue, vocab=residue_symbol2id) for residue in seq]
            # # pad it with zeros while sequence is short
            # while len(_oh_residue) < max_len:
            #     _oh_residue.append([0] * len(residue_symbol2id))
            _oh_residue_feature.append(torch.FloatTensor(_oh_residue))  # n * d
        residue_feature = torch.stack(_oh_residue_feature, dim=0)  # N * n * d

    # call PLM to generate residue feature
    elif full_residue_feat:
        possible_residue_file_path_pretrained = os.path.join(
            feature_folder, "full_protein_features_%s_%s_%s.pt" % (args.dataset, feature_generator, args.split)
        )
        possible_residue_file_path = os.path.join(
            feature_folder, "full_protein_features_%s_ESM-1b.pt" % args.dataset
        )
        if os.path.exists(possible_residue_file_path_pretrained):
            residue_feature = torch.load(f=possible_residue_file_path_pretrained)
            logger.warning("Loaded residue feature at %s" % possible_residue_file_path_pretrained)
        elif os.path.exists(possible_residue_file_path):
            residue_feature = torch.load(f=possible_residue_file_path)
            logger.warning("Loaded residue feature at %s" % possible_residue_file_path)
        else:
            if "ESM" in feature_generator:
                llm = tg_model.EvolutionaryScaleModeling(
                    path=esm_folder, model=feature_generator, readout="mean"
                ).to(args.device)
            else:
                llm = tg_model.EvolutionaryScaleModeling(
                    path=esm_folder, model="ESM-1b", readout="mean"
                ).to(args.device)
            loaded = False

            if pretrained:
                file_path = os.path.join(
                    model_param_folder,
                    "%s-%s/Llm_Mlp_%s.pth" % (args.dataset, feature_generator, args.split)
                )
                if os.path.exists(file_path):
                    logger.warning("loading model parameters at %s" % file_path)
                    state = torch.load(file_path, map_location='cpu')
                    if state['Llm_model']['mapping'].size != llm.state_dict()['mapping'].size:
                        state['Llm_model']['mapping'] = llm.state_dict()['mapping']
                    llm.load_state_dict(state["Llm_model"])
                    logger.warning("LLM model parameters loaded")
                    loaded = True
                else:
                    logger.warning("There is no fine-tuned parameters, use original parameters.")
            else:
                logger.warning("Use %s model parameters" % feature_generator)

            llm.eval()
            _pretrained_residue_feature = []
            # indices_list = slice_indices(indices=range(len(sequences)), batch_size=args.batch_size)
            indices_list = slice_indices(indices=range(len(sequences)), batch_size=128)
            if pretrained and loaded:
                indices_list = tqdm(indices_list,
                                    "Loading residue pretrained embedding from fine-tuned %s" % feature_generator)
            else:
                if "ESM" in feature_generator:
                    indices_list = tqdm(indices_list,
                                        "Loading residue pretrained embedding from %s" % feature_generator)
                else:
                    indices_list = tqdm(indices_list,
                                        "Loading residue pretrained embedding from %s" % "ESM-1b")
            for _, indices in enumerate(indices_list):
                batch_sequences = [sequences[index] for index in indices]
                batch_proteins = td_data.PackedProtein.from_sequence(
                    batch_sequences, atom_feature=None, bond_feature=None, residue_feature="default"
                ).to(args.device)
                batch_emb = llm(
                    graph=batch_proteins, input=batch_proteins.node_feature.float(), all_loss=None, metric=None
                )
                _pretrained_residue_feature.append(batch_emb["residue_feature"].detach().cpu())
            if "ESM" in feature_generator:
                residue_feature = torch.cat(_pretrained_residue_feature, dim=0).reshape(
                        [len(sequences), -1, output_dim[args.feature_generator]])
            else:
                residue_feature = torch.cat(_pretrained_residue_feature, dim=0).reshape(
                    [len(sequences), -1, output_dim["ESM-1b"]])

            del llm
            del batch_proteins
            torch.cuda.empty_cache()

    elif args.light_residue_feat:
        if pretrained:
            # residue_file_path = os.path.join(
            #     feature_folder, "residue_features_%s_%s_%s.pt" % (args.dataset, feature_generator, args.split)
            # )
            residue_file_path = os.path.join(
                feature_folder, "protein_features_%s_%s_%s.pt" % (args.dataset, feature_generator, args.split)
            )
        else:
            # residue_file_path = os.path.join(
            #     feature_folder, "residue_features_%s_%s.pt" % (args.dataset, feature_generator)
            # )
            residue_file_path = os.path.join(
                feature_folder, "protein_features_%s_%s.pt" % (args.dataset, feature_generator)
            )
        residue_feature = torch.load(f=residue_file_path)
        logger.warning("Loaded residue feature at %s" % residue_file_path)
    
    else:
        logger.warning("No residue feature")
        residue_feature = None

    return protein_feature, residue_feature


def load_similarity_matrix(similarity, dataset, logger):
    if similarity == "gzip":
        # file_path = os.path.join(similarity_folder, "gzip_similarity/%s.npy" % dataset)
        file_path = os.path.join(similarity_folder, "gzip_similarity/%s.npz" % dataset)
        if os.path.exists(file_path):
            logger.warning("loading similarity matrix at %s" % file_path)
            # similarities = torch.FloatTensor(np.load(file=file_path))
            similarities = torch.FloatTensor(np.load(file=file_path)['matrix'])
        else:
            raise ValueError("gzip similarity matrix file does not exist at %s" % file_path)
    elif similarity == "nw":
        # file_path = os.path.join(similarity_folder, "Needleman-Wunsch_similarity/%s.npy" % dataset)
        file_path = os.path.join(similarity_folder, "Needleman-Wunsch_similarity/%s.npz" % dataset)
        if os.path.exists(file_path):
            logger.warning("loading similarity matrix at %s" % file_path)
            # similarities = torch.FloatTensor(np.load(file=file_path))
            similarities = torch.FloatTensor(np.load(file=file_path)['matrix'])
        else:
            raise ValueError("Needleman-Wunsch similarity matrix file does not exist at %s" % file_path)
    else:
        raise ValueError("Similarity wrong")

    return similarities
