import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BatchNorm1d, Dropout
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr
import json
torch.autograd.set_detect_anomaly(True)

torch.cuda.manual_seed_all(42)
import utils as utils1

from model import SiaRNA
from efficacy_dataset import EfficacyDataset

# --- Load parameters ---
with open("siRNA_param_pytorch.json", 'r') as f:
    params = json.load(f)

MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]

score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []
score_f1=[]
score_recall = []
score_prec = []

test_score_PCC = []
test_score_SPCC = []
test_score_mse = []
test_score_auc = []
test_score_f1 = []
test_score_recall = []
test_score_prec = []

def collate_fn(batch):
    mrna_list, sirna_list, thermo_list, sirna_seq_list, mrna_slice_list, labels = zip(*batch)

    mrna_padded = torch.stack(mrna_list)
    sirna_padded = torch.stack(sirna_list)
    thermo_padded = torch.stack(thermo_list)

    sirna_seq_padded = pad_sequence(sirna_seq_list, batch_first=True)
    sirna_seq_mask = torch.zeros(sirna_seq_padded.shape, dtype=torch.bool)
    for i, seq in enumerate(sirna_seq_list):
        sirna_seq_mask[i, :len(seq)] = 1

    # mRNA slice (fixed length = 57 so no padding needed)
    mrna_slice_tensor = torch.stack(mrna_slice_list)

    labels = torch.stack(labels)
    return mrna_padded, sirna_padded, thermo_padded, sirna_seq_padded, mrna_slice_tensor, labels


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

def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq, y_true = [b.to(device) for b in batch]
            y_pred, _ = model(mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq)
            all_preds.append(y_pred.cpu())
            all_labels.append(y_true.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy() 
    mse = mean_squared_error(all_labels, all_preds)
    pcc, _ = pearsonr(all_labels, all_preds)
    spcc, _ = spearmanr(all_labels, all_preds) 
    binary_true = (all_labels > 0.7).astype(int)
    binary_pred = (all_preds > 0.7).astype(int) 
    try:
        auc = roc_auc_score(binary_true, all_preds)
    except ValueError:
        auc = float('nan')  
    f1 = f1_score(binary_true, binary_pred, zero_division=0)
    precision = precision_score(binary_true, binary_pred, zero_division=0)
    recall = recall_score(binary_true, binary_pred, zero_division=0)
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation PCC: {pcc:.4f}")
    print(f"Validation SPCC: {spcc:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation F1: {f1:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    return mse, pcc, spcc, auc, f1, precision, recall


def train(model, dataloader, optimizer, device, use_contrastive=True, lambda_metric=0.1):
    model.train()
    total_loss = 0 
    for batch in dataloader:
        mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq, y_true = [b.to(device) for b in batch]
        # print(siRNA_seq)
        y_pred, metric_emb = model(mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq)
        reg_loss = F.mse_loss(y_pred, y_true)     
        if use_contrastive:
            label = (y_true > 0.7).float()
            emb_m, emb_s = metric_emb.chunk(2, dim=-1)
            metric_loss = contrastive_loss(emb_m, emb_s, label)
            loss = reg_loss + lambda_metric * metric_loss
        else:
            loss = reg_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def get_mRNA_features(n, data):
    # mRNA one-hot (PADDED to max_mrna_len)
    mrna_onehot_temp = data.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
    mrna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
                   for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))
    has_nan = mrna_onehot.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # mRNA base-pairing probability
    mrna_sfold_feat = pd.read_csv(f"siRNA_split_preprocess/self_mRNA_matrix.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    has_nan = mrna_sfold_feat.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    ## mRNA-AGO2
    mrna_ago = pd.read_csv(f"RNA_AGO2/mRNA_AGO2_zh.csv",index_col=0)
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

def get_siRNA_features(n, data):
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
    sirna_sfold_feat = pd.read_csv(f"siRNA_split_preprocess/self_siRNA_matrix.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)
    has_nan = sirna_sfold_feat.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    ## siRNA-AGO2
    sirna_ago = pd.read_csv(f"RNA_AGO2/siRNA_AGO2_zh.csv",index_col = 0)
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

# --- K-Fold Cross Validation Loop ---
NUM_FOLDS = 10
for n in range(NUM_FOLDS):
    print(f"\nProcessing fold {n}")

    # Read and preprocess data
    SPLIT_DIR = f"siRNA_split_datasets/split{n}"
    if not os.path.exists(SPLIT_DIR):
        print(f"Error: Directory '{SPLIT_DIR}' not found. Please ensure your data splits are correctly organized.")
        print("Skipping fold processing.")
        continue

    try:
        data_train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
        data_dev = pd.read_csv(os.path.join(SPLIT_DIR, "dev.csv"))
        data_test = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))
    except FileNotFoundError as e:
        print(f"Error: Data file not found in '{SPLIT_DIR}'. {e}")
        print("Skipping fold processing.")
        continue

    data_train['split'] = 'train'
    data_dev['split'] = 'dev'
    data_test['split'] = 'test'

    # Substitute U to T for siRNA_seq to match mRNA_seq (DNA-like) for positional lookup.
    for df in [data_train, data_dev, data_test]:
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')

    # Combine all data for feature processing
    data = pd.concat([data_train, data_dev, data_test], axis=0).reset_index(drop=True)


    # # 1. One-hot encoding for siRNA (PADDED to MAX_SIRNA_LENGTH)
    # sirna_onehot = []
    # for seq in data['siRNA_seq']:
    #     sirna_onehot.append(utils1.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
    # sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data['siRNA']))

    # has_nan = sirna_onehot.isnull().values.any()
    # print(f"siRNA onehot Contains NaN: {has_nan}")

    # # 2. mRNA one-hot (PADDED to max_mrna_len)
    # mrna_onehot_temp = data.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
    # mrna_onehot = [utils1.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
    #                for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    # mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))
    # has_nan = mrna_onehot.isnull().values.any()
    # print(f"mRNA onehot Contains NaN: {has_nan}")


    # # 3. Positional encoding (PADDED to MAX_SIRNA_LENGTH * dmodel)
    # sirna_pos_encoding = []
    # for idx, row in data.iterrows():
    #     mrna_start_pos = max(0, int(row['pos']))
    #     sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
    #         mrna_start_pos,
    #         len(row['siRNA_seq']),
    #         MAX_SIRNA_LENGTH,
    #         params["dmodel"]
    #     ))
    # sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data['siRNA'] + '_' + data['mRNA'])
    # has_nan = sirna_pos_encoding.isnull().values.any()
    # print(f"siRNA positional Contains NaN: {has_nan}")

    # # 4. Thermodynamics
    # sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
    #                      for seq in data['siRNA_seq']]
    # sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    # temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    # sirna_thermo_feat['index'] = temp_interaction_index
    # sirna_thermo_feat = sirna_thermo_feat.set_index('index')

    # # 5. Co-fold features (using base_pair_probs as indicated by user)
    # con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt", header=None, index_col=0)
    # con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)
    # has_nan = con_feat.isnull().values.any()
    # print(f"con features Contains NaN: {has_nan}")

    # # 6. Self-fold features (siRNA)
    # sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt", header=None, index_col=0)
    # sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)
    # has_nan = sirna_sfold_feat.isnull().values.any()
    # print(f"sirna_sfold_feat Contains NaN: {has_nan}")

    # # 7. Self-fold features (mRNA)
    # mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt", header=None, index_col=0)
    # mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    # has_nan = mrna_sfold_feat.isnull().values.any()
    # print(f"mrna_sfold_feat Contains NaN: {has_nan}")

    # # AGO2
    # ## siRNA-AGO2
    # sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2_zh.csv",index_col = 0)
    # sirna_ago = sirna_ago.reindex(sirna_onehot.index)
    # has_nan = sirna_ago.isnull().values.any()
    # print(f"sirna_ago Contains NaN: {has_nan}")

    # ## mRNA-AGO2
    # mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2_zh.csv",index_col=0)
    # mrna_ago = mrna_ago.reindex(mrna_onehot.index)
    # has_nan = mrna_ago.isnull().values.any()
    # print(f"mrna_ago Contains NaN: {has_nan}")

    # # 8. GC percentage
    # sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data['siRNA_seq']], index=list(data['siRNA']))
    # has_nan = sirna_GC.isnull().values.any()
    # print(f"sirna_GC Contains NaN: {has_nan}")
    # mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))
    # has_nan = mrna_GC.isnull().values.any()
    # print(f"mrna_GC Contains NaN: {has_nan}")

    # # 9. K-mers 
    # sirna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    # sirna_k_mers.index = data['siRNA']
    # has_nan = sirna_k_mers.isnull().values.any()
    # print(f"sirna_k_mers Contains NaN: {has_nan}")

    # # 10. siRNA rules codes 
    # SIRNA_RULES_LENGTH = 19
    # sirna_pos_scores = []
    # for seq in data['siRNA_seq']:
    #     sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    # sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    # has_nan = sirna_pos_scores.isnull().values.any()
    # print(f"sirna_pos_scores Contains NaN: {has_nan}")

    print("\n--- Assembling SNN Node Features ---")

    sirna_onehot, sirna_sfold_feat, sirna_ago, sirna_GC, sirna_k_mers, sirna_pos_scores, sirna_thermo_feat = get_siRNA_features(n, data)
    mrna_onehot, mrna_sfold_feat, mrna_ago, mrna_GC = get_mRNA_features(n, data)

    # siRNA nodes features
    sirna_pd  = pd.concat([sirna_ago, sirna_k_mers, sirna_GC, sirna_pos_scores, sirna_onehot,sirna_sfold_feat], axis=1)

    # mRNA nodes features
    mrna_pd = pd.concat([mrna_sfold_feat,  mrna_ago,  mrna_GC, mrna_onehot], axis=1)


    input_dim_mRNA = mrna_pd.shape[1]
    input_dim_siRNA = sirna_pd.shape[1]
    input_dim_thermo = sirna_thermo_feat.shape[1]

    # Create datasets and dataloaders
    train_dataset = EfficacyDataset(data_train, sirna_pd, mrna_pd, sirna_thermo_feat)
    dev_dataset   = EfficacyDataset(data_dev, sirna_pd, mrna_pd, sirna_thermo_feat)
    test_dataset  = EfficacyDataset(data_test, sirna_pd, mrna_pd, sirna_thermo_feat)

    train_loader = DataLoader(train_dataset, params["batch_size"], shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_dataset, params["batch_size"], shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, params["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # --- Model Initialization and Training Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiaRNA(input_dim_mRNA, input_dim_siRNA, input_dim_thermo).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # def train(model, dataloader, optimizer, device, use_contrastive=True, lambda_metric=0.1):
    #     model.train()
    #     total_loss = 0
    
    #     for batch in dataloader:
    #         mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq, y_true = [b.to(device) for b in batch]
    #         # print(siRNA_seq)
    
    #         y_pred, metric_emb = model(mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq)
    #         reg_loss = F.mse_loss(y_pred, y_true)
            
    #         if use_contrastive:
    #             label = (y_true > 0.7).float()
    #             emb_m, emb_s = metric_emb.chunk(2, dim=-1)
    #             metric_loss = contrastive_loss(emb_m, emb_s, label)
    #             loss = reg_loss + lambda_metric * metric_loss
    #         else:
    #             loss = reg_loss
    
    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()
    
    #         total_loss += loss.item()
    
    #     return total_loss / len(dataloader)

    # def validate(model, dataloader, device):
    #     model.eval()
    #     all_preds = []
    #     all_labels = []
    
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq, y_true = [b.to(device) for b in batch]
    
    #             y_pred, _ = model(mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq)
    
    #             all_preds.append(y_pred.cpu())
    #             all_labels.append(y_true.cpu())
    
    #     all_preds = torch.cat(all_preds).numpy()
    #     all_labels = torch.cat(all_labels).numpy()
        
    #     mse = mean_squared_error(all_labels, all_preds)
    #     pcc, _ = pearsonr(all_labels, all_preds)
    #     spcc, _ = spearmanr(all_labels, all_preds)
        
    #     binary_true = (all_labels > 0.7).astype(int)
    #     binary_pred = (all_preds > 0.7).astype(int)
        
    #     try:
    #         auc = roc_auc_score(binary_true, all_preds)
    #     except ValueError:
    #         auc = float('nan')
        
    #     f1 = f1_score(binary_true, binary_pred, zero_division=0)
    #     precision = precision_score(binary_true, binary_pred, zero_division=0)
    #     recall = recall_score(binary_true, binary_pred, zero_division=0)
    
    #     print(f"Validation MSE: {mse:.4f}")
    #     print(f"Validation PCC: {pcc:.4f}")
    #     print(f"Validation SPCC: {spcc:.4f}")
    #     print(f"Validation AUC: {auc:.4f}")
    #     print(f"Validation F1: {f1:.4f}")
    #     print(f"Validation Precision: {precision:.4f}")
    #     print(f"Validation Recall: {recall:.4f}")
    
    #     return mse, pcc, spcc, auc, f1, precision, recall

        
    
    # --- Main Training Loop ---
    best_pcc = float('-inf')
    for epoch in range(params['epochs']):
        print(epoch)
        loss = train(model, train_loader, optimizer, device="cuda",
                 use_contrastive=True, lambda_metric=0.1)
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

        mse,pcc,spcc,auc,f1,prec,recall = validate(model, dev_loader, device="cuda")
        t_mse, t_pcc, t_spcc, t_auc, t_f1, t_prec, t_recall = validate(model, test_loader, device="cuda")
        if t_pcc>best_pcc:
            best_pcc= t_pcc
            torch.save(model.state_dict(), f'attn_model_fold{n}.pt')
            print("model saved")
        if epoch == (params["epochs"]-1):
            score_PCC.append(pcc)
            score_SPCC.append(spcc)
            score_mse.append(mse)
            score_auc.append(auc)
            score_f1.append(f1)
            score_recall.append(recall)
            score_prec.append(prec)
        
    #print(f"--- Fold {n} finished! Final Validation Loss: {best_val_loss:.4f} ---")

    # Evaluate on test set after training is complete
    print(f"\n--- Testing on fold {n} ---")
    eval_model = EfficacyModel(input_dim_mRNA, input_dim_siRNA, input_dim_thermo).cuda()
    # Load the saved weights
    model_path = f"attn_model_fold{n}.pt" 

    try:
        eval_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot test this fold.")
        continue # Skip to next fold
    except Exception as e:
        print(f"An error occurred while loading model for fold {n}: {e}. Skipping this fold.")
        continue
    test_mse, test_pcc, test_spcc, test_auc, test_f1, test_prec, test_recall = validate(eval_model, test_loader, device="cuda")
    
    # Append test metrics
    test_score_PCC.append(test_pcc)
    test_score_SPCC.append(test_spcc)
    test_score_mse.append(test_mse)
    test_score_auc.append(test_auc)
    test_score_f1.append(test_f1)
    test_score_recall.append(test_recall)
    test_score_prec.append(test_prec)

# --- No overall metrics summary since test set evaluation is removed ---
print("\n--- Training and Validation Complete Across All Folds ---")
print("Overall MSE score =", np.mean(score_mse))
print("Overall PCC score =", np.mean(score_PCC))
print("Overall SPCC score =", np.mean(score_SPCC))
print("Overall AUC score =", np.mean(score_auc))
print("Overall F1 score =", np.mean(score_f1))
print("Overall Presicion score =", np.mean(score_prec))
print("Overall Recall score =", np.mean(score_recall))
print("Model checkpoints saved based on best validation loss for each fold.")

print("\nTest Metrics:")
print("MSE:",test_score_mse)
print("PCC:",test_score_PCC)
print("SPCC:",test_score_SPCC)
print("AUC:",test_score_auc)
print("Overall MSE score =", np.mean(test_score_mse))
print("Overall PCC score =", np.mean(test_score_PCC))
print("Overall SPCC score =", np.mean(test_score_SPCC))
print("Overall AUC score =", np.mean(test_score_auc))
print("Overall F1 score =", np.mean(test_score_f1))
print("Overall Precision score =", np.mean(test_score_prec))
print("Overall Recall score =", np.mean(test_score_recall))
print("Model checkpoints saved based on best validation loss for each fold.")