#single value inference
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import json
import re
import sys
import subprocess
import os
import shutil
import argparse
import utils as utils # Features calculation
import preprocess
import predict_RPI
import warnings
warnings.filterwarnings("ignore")
from model import SiaRNA
from efficacy_dataset import EfficacyDataset

# Load parameters
params = json.load(open("./siRNA_param_pytorch.json", 'r'))

MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]
INPUT_DIR = "data/input/"
MODEL_PATH = f"checkpoints/best_model_fold.pt"


def collate_fn(batch):
    mrna_list, sirna_list, thermo_list, sirna_seq_list, mrna_slice_list = zip(*batch)

    mrna_padded = torch.stack(mrna_list)
    sirna_padded = torch.stack(sirna_list)
    thermo_padded = torch.stack(thermo_list)

    sirna_seq_padded = pad_sequence(sirna_seq_list, batch_first=True)
    sirna_seq_mask = torch.zeros(sirna_seq_padded.shape, dtype=torch.bool)
    for i, seq in enumerate(sirna_seq_list):
        sirna_seq_mask[i, :len(seq)] = 1

    mrna_slice_tensor = torch.stack(mrna_slice_list)

    return mrna_padded, sirna_padded, thermo_padded, sirna_seq_padded, mrna_slice_tensor


def contrastive_loss(emb_m, emb_s, label, margin=1.0):
    dist = torch.norm(emb_m - emb_s, p=2, dim=1)
    if torch.isnan(dist).any():
        print("NaN detected in distance.")
    relu_term = F.relu(margin - dist)
    if torch.isnan(relu_term).any():
        print("NaN detected in relu term.")
    loss = label * dist.pow(2) + (1 - label) * relu_term.pow(2)
    if torch.isnan(loss).any():
        print("NaN detected in loss.")
    return loss.mean()

def get_mRNA_features(reqId, data):
    # mRNA one-hot (PADDED to max_mrna_len)
    mrna_onehot_temp = data.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
    mrna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
                   for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))
    has_nan = mrna_onehot.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # mRNA base-pairing probability
    mrna_sfold_feat = pd.read_csv(f"data/input/{reqId}_preprocess/{reqId}_con_matrix_meanSum100.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    has_nan = mrna_sfold_feat.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    ## mRNA-AGO2
    mrna_ago = pd.read_csv(f"data/input/{reqId}_RNA_AGO2/{reqId}_mRNA_AGO2_zh.csv",index_col=0)
    mrna_ago = mrna_ago.reindex(mrna_onehot.index)
    has_nan = mrna_ago.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # mRNA GC content
    mrna_GC = pd.DataFrame([utils.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))
    has_nan = mrna_GC.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    return mrna_onehot, mrna_sfold_feat, mrna_ago, mrna_GC

def get_siRNA_features(reqId, data):
    #siRNA one hot encoding
    sirna_onehot = []
    for seq in data['siRNA_seq']:
        sirna_onehot.append(utils.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data['siRNA']))

    has_nan = sirna_onehot.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing") 
        
    #siRNA thermodynamic features
    sirna_thermo_feat = [utils.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                         for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index')

    # siRNA base-pairing probabilities
    sirna_sfold_feat = pd.read_csv(f"data/input/{reqId}_preprocess/{reqId}_self_siRNA_matrix_meanSum6.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)
    has_nan = sirna_sfold_feat.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    ## siRNA-AGO2
    sirna_ago = pd.read_csv(f"data/input/{reqId}_RNA_AGO2/{reqId}_siRNA_AGO2_zh.csv",index_col = 0)
    sirna_ago = sirna_ago.reindex(sirna_onehot.index)
    has_nan = sirna_ago.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    #siRNA GC content
    sirna_GC = pd.DataFrame([utils.countGC(seq) for seq in data['siRNA_seq']], index=list(data['siRNA']))
    has_nan = sirna_GC.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    #K-mers
    sirna_1_mer = pd.DataFrame([utils.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data['siRNA']
    has_nan = sirna_k_mers.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # siRNA rules codes
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data['siRNA_seq']:
        sirna_pos_scores.append(utils.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    has_nan = sirna_pos_scores.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    return sirna_onehot, sirna_sfold_feat, sirna_ago, sirna_GC, sirna_k_mers, sirna_pos_scores, sirna_thermo_feat
        

@torch.no_grad()
def inference(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader: 
            mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq = [b.to(device) for b in batch]
            y_pred, _ = model(mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq)
            all_preds.append(y_pred.cpu())
    all_preds = torch.cat(all_preds)
    return all_preds

def main():
    parser = argparse.ArgumentParser(description="siRNa-mRNA efficacy prediction")
    parser.add_argument("-r", "--request_id", required=True, help="Request Id")
    parser.add_argument("-m", "--mRNA_seq_file", required=True, help="Path to the mRNA sequence file (FASTA format)")
    parser.add_argument("-s", "--siRNA_seq_file", required=True, help="Path to the siRNA sequence file (FASTA format)")
    args = parser.parse_args()
    reqId=args.request_id
    mRNA_path=args.mRNA_seq_file
    siRNA_path=args.siRNA_seq_file
    
    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        # print(f"Directory '{folder_path}' ensured (created or already exists).")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    try:
        preprocess.main(reqId, mRNA_path, siRNA_path)
    except:
        raise ValueError(f"Preprocessing failed")
    main_dir = os.getcwd()
    try:
        predict_RPI.main("mRNA", "test", mRNA_path, f"{reqId}")
        predict_RPI.main("siRNA", "test", siRNA_path, f"{reqId}")
    except:
        raise ValueError("RPI predict failed")
    os.chdir(main_dir)
    # print("main entered")
    try:
        TEST_FILE = f"data/input/{reqId}_inference_siRNA_mRNA_pairs.csv"
        data = pd.read_csv(TEST_FILE)
        # print(f"Successfully loaded test data from: {test_file}")
    except FileNotFoundError:
        raise FileNotFoundError
    except Exception as e:
        raise FileNotFoundError

    # Substitue U to T in siRNA sequence
    data['siRNA_seq'] = data['siRNA_seq'].replace('U', 'T', regex=True)
    
    '''

    Features for SNN

    '''

    sirna_onehot, sirna_sfold_feat, sirna_ago, sirna_GC, sirna_k_mers, sirna_pos_scores, sirna_thermo_feat = get_siRNA_features(reqId, data)
    mrna_onehot, mrna_sfold_feat, mrna_ago, mrna_GC = get_mRNA_features(reqId, data)

    # The features of SNN nodes
    sirna_pd  = pd.concat([sirna_ago, sirna_k_mers, sirna_GC, sirna_pos_scores, sirna_onehot,sirna_sfold_feat], axis=1)

    mrna_pd = pd.concat([mrna_sfold_feat,  mrna_ago,  mrna_GC, mrna_onehot], axis=1)

    input_dim_mRNA = mrna_pd.shape[1]
    input_dim_siRNA = sirna_pd.shape[1]
    input_dim_thermo = sirna_thermo_feat.shape[1]     

    dataset = EfficacyDataset(data, sirna_pd, mrna_pd, sirna_thermo_feat)
    data_loader = DataLoader(dataset, params["batch_size"], shuffle = False, collate_fn = collate_fn)
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiaRNA(input_dim_mRNA, input_dim_siRNA, input_dim_thermo).cuda()

    # 2. Load the saved weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Cannot test this fold.")
        raise FileNotFoundError
    except Exception as e:
        print(f"An error occurred while loading model for fold {n}: {e}. Skipping this fold.")
        raise ValueError

    # Setting the model for inference
    model.eval()
        
    inference_preds = inference(model, data_loader, device) 

    # Convert to numpy array
    inference_preds = inference_preds.numpy().flatten()

    print("Predictions:", inference_preds[0])


    prefixes = [f"{reqId}_RNAcofold", f"{reqId}_RNAfold"]
    for folder in os.listdir(INPUT_DIR):
        folder_path = os.path.join(INPUT_DIR, folder)
        if os.path.isdir(folder_path) and any(folder.startswith(prefix) for prefix in prefixes):
            shutil.rmtree(folder_path)
        
    return inference_preds[0]
    
if __name__ == "__main__":
    main()
    


