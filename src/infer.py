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
from flask import request
import subprocess
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import src.utils as utils # Features calculation
sys.path.append('src')
#print('main directory', os.getcwd())
from predict_RPI import main as main_ago
from preprocess import main as main_bpp


# Load parameters
params = json.load(open("src/siRNA_param_pytorch.json", 'r'))

# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj   = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key_value, mask=None):
        """
        query:     (B, q_len, embed_dim) - e.g. 57-nt mRNA slice embeddings
        key_value: (B, kv_len, embed_dim) - e.g. siRNA sequence embeddings
        mask:      (B, kv_len) - optional attention mask for key_value
        """
        B, q_len, _ = query.size()
        kv_len = key_value.size(1)
        
        # Project to Q, K, V
        Q = self.query_proj(query)        # (B, q_len, embed_dim)
        K = self.key_proj(key_value)      # (B, kv_len, embed_dim)
        V = self.value_proj(key_value)    # (B, kv_len, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, q_len, head_dim)
        K = K.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, kv_len, head_dim)
        V = V.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, kv_len, head_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, q_len, kv_len)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)                   # (B, 1, 1, kv_len)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax over kv_len
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, q_len, kv_len)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (B, num_heads, q_len, head_dim)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(B, q_len, self.embed_dim)  # (B, q_len, embed_dim)
        
        # Output projection
        output = self.out_proj(attended)  # (B, q_len, embed_dim)
        
        return output


# ------------------------
# 1. Sequence Encoder
# ------------------------
class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, feature_dim)
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h  # shape: (batch_size, embed_dim)




class EfficacyModel(nn.Module):
    def __init__(self, input_dim_mRNA, input_dim_siRNA, input_dim_thermo,
                 proj_dim=256, hidden_dim=64, embed_dim=32, fusion_dim=128, mlp_dim=64):
        super().__init__()
        
        # Existing components
        self.proj_mRNA = nn.Linear(input_dim_mRNA, proj_dim)
        self.proj_siRNA = nn.Linear(input_dim_siRNA, proj_dim)
        self.ln_proj = nn.LayerNorm(proj_dim)
        self.shared_encoder = SequenceEncoder(proj_dim, hidden_dim, embed_dim)
        
        # NEW: Sequence embedding for siRNA sequences  
        self.nucleotide_embedding = nn.Embedding(5, embed_dim)  # A, T, G, C/U, N -> embed_dim
        self.sequence_proj = nn.Linear(embed_dim, embed_dim)
        
        # Cross-attention
        self.cross_attention = CrossAttention(embed_dim, num_heads=4)
        
        # Fusion and MLP (same as before)
        self.fc_fusion = nn.Linear(embed_dim * 2, fusion_dim)
        self.ln_fusion = nn.LayerNorm(fusion_dim)
        self.fc_mlp1 = nn.Linear(fusion_dim + input_dim_thermo + embed_dim, mlp_dim)
        self.ln_mlp1 = nn.LayerNorm(mlp_dim)
        self.fc_out = nn.Linear(mlp_dim, 1)

    def forward(self, mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_slice):
        mRNA_proj = self.ln_proj(self.proj_mRNA(mRNA_x))
        siRNA_proj = self.ln_proj(self.proj_siRNA(siRNA_x))

        mRNA_hidden = self.shared_encoder(mRNA_proj)
        siRNA_hidden = self.shared_encoder(siRNA_proj)

        # Embed sequences
        siRNA_seq_embedded = self.sequence_proj(self.nucleotide_embedding(siRNA_seq))
        mRNA_slice_embedded = self.sequence_proj(self.nucleotide_embedding(mRNA_slice))  

        # Cross-attention: query = mRNA slice, key/value = siRNA sequence
        cross_attended1 = self.cross_attention(
            query=mRNA_slice_embedded,   
            key_value=siRNA_seq_embedded 
        ) 

        # Cross-attention: key/value = mRNA slice, query = siRNA sequence
        cross_attended2 = self.cross_attention(
            query = siRNA_seq_embedded,
            key_value = mRNA_slice_embedded
        )

        # Pool the cross-attended slice (mean-pooling)
        cross_attended1 = cross_attended1.mean(dim=1)  # (B,embed_dim)
        cross_attended2 = cross_attended2.mean(dim=1)
        cross_attended = torch.stack([cross_attended1, cross_attended2], dim=0).mean(dim=0)  # (B,embed_dim)

        fusion = torch.cat([mRNA_hidden, siRNA_hidden], dim=-1)
        h = self.ln_fusion(F.relu(self.fc_fusion(fusion)))

        h_thermo = torch.cat([h, cross_attended, thermo_x], dim=-1)
        h_mlp = self.ln_mlp1(F.relu(self.fc_mlp1(h_thermo)))
        out = self.fc_out(h_mlp)

        return out.squeeze(-1), h


class EfficacyDataset(Dataset):
    def __init__(self, df, sirna_pd, mrna_pd, thermo_pd):
        self.df = df.reset_index(drop=True)
        self.sirna_pd = sirna_pd
        self.mrna_pd = mrna_pd
        self.thermo_pd = thermo_pd

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sirna_id = row["siRNA"].strip()
        mrna_id = row["mRNA"].strip()
        pair_id = f"{sirna_id}_{mrna_id}"

        sirna_x = torch.tensor(self.sirna_pd.loc[sirna_id].values, dtype=torch.float32)
        mrna_x  = torch.tensor(self.mrna_pd.loc[mrna_id].values, dtype=torch.float32)
        thermo_x = torch.tensor(self.thermo_pd.loc[pair_id].values, dtype=torch.float32)

        # siRNA sequence
        sirna_seq_tensor = self.sequence_to_tensor(row["siRNA_seq"])

        # NEW: mRNA slice (57nt around pos)
        mrna_seq = row["mRNA_seq"]
        pos = int(row["pos"])
        start = max(0, pos - 28)
        end = min(len(mrna_seq), pos + 29)
        mrna_slice = mrna_seq[start:end]

        # pad if <57
        if len(mrna_slice) < 57:
            mrna_slice = mrna_slice + "N" * (57 - len(mrna_slice))



        mrna_slice_tensor = self.sequence_to_tensor(mrna_slice)

        return mrna_x, sirna_x, thermo_x, sirna_seq_tensor, mrna_slice_tensor

    def sequence_to_tensor(self, sequence):
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'U': 1, 'N': 4}
        return torch.tensor([nucleotide_map.get(nt.upper(), 0) for nt in sequence], dtype=torch.long)



def collate_fn(batch):
    mrna_list, sirna_list, thermo_list, sirna_seq_list, mrna_slice_list = zip(*batch)

    mrna_padded = torch.stack(mrna_list)
    sirna_padded = torch.stack(sirna_list)
    thermo_padded = torch.stack(thermo_list)

    # siRNA sequence padding
    sirna_seq_padded = pad_sequence(sirna_seq_list, batch_first=True)
    sirna_seq_mask = torch.zeros(sirna_seq_padded.shape, dtype=torch.bool)
    for i, seq in enumerate(sirna_seq_list):
        sirna_seq_mask[i, :len(seq)] = 1

    # mRNA slice (fixed length = 57 so no padding needed)
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

def main():
    reqId=request.form.get('reqId')
    # print(reqID)
    #reqId="1234"
    print("Preprocessing........")
    # subprocess.run(["python3", "src/preprocess.py",reqId])
    try:
        main_bpp(reqId)
    except:
        raise ValueError(f"Preprocessing failed")
    main_dir = os.getcwd()
    print("cwd", main_dir)
    # os.chdir('src')
    try:
        main_ago("mRNA", "test", f"{reqId}")
        main_ago("siRNA", "test", f"{reqId}")
    except:
        raise ValueError("RPI predict failed")
    # os.chdir(main_dir)
    print("main entered")
    try:
        test_file = f"src/data/input/{reqId}_inference_siRNA_mRNA_pairs.csv"
        data = pd.read_csv(test_file)
        print(f"Successfully loaded test data from: {test_file}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load test data: {e}")

    # Substitue U to T in siRNA sequence
    data['siRNA_seq'] = data['siRNA_seq'].replace('U', 'T', regex=True)
    
    '''

    Feature processing

    '''
    #------------ one-hot encoding --------------
    sirna_onehot = []
    for seq in data['siRNA_seq']:
        sirna_onehot.append(utils.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data['siRNA']))

    has_nan = sirna_onehot.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # 2. mRNA one-hot (PADDED to max_mrna_len)
    mrna_onehot_temp = data.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
    mrna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
                   for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))
    has_nan = mrna_onehot.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")
    


    # 4. Thermodynamics (PADDED to MAX_SIRNA_LENGTH)
    sirna_thermo_feat = [utils.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                         for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index')

    # 6. Self-fold features (siRNA)
    sirna_sfold_feat = pd.read_csv(f"src/data/input/{reqId}_preprocess/{reqId}_self_siRNA_matrix_meanSum6.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)
    has_nan = sirna_sfold_feat.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # 7. Self-fold features (mRNA)
    mrna_sfold_feat = pd.read_csv(f"src/data/input/{reqId}_preprocess/{reqId}_con_matrix_meanSum100.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    has_nan = mrna_sfold_feat.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # AGO2
    ## siRNA-AGO2
    sirna_ago = pd.read_csv(f"src/data/input/{reqId}_RNA_AGO2/{reqId}_siRNA_AGO2_zh.csv",index_col = 0)
    sirna_ago = sirna_ago.reindex(sirna_onehot.index)
    has_nan = sirna_ago.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    ## mRNA-AGO2
    mrna_ago = pd.read_csv(f"src/data/input/{reqId}_RNA_AGO2/{reqId}_mRNA_AGO2_zh.csv",index_col=0)
    mrna_ago = mrna_ago.reindex(mrna_onehot.index)
    has_nan = mrna_ago.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")
    # AGO2
    ## siRNA-AGO2

    # 8. GC percentage (variable length robust)
    sirna_GC = pd.DataFrame([utils.countGC(seq) for seq in data['siRNA_seq']], index=list(data['siRNA']))
    has_nan = sirna_GC.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")
    mrna_GC = pd.DataFrame([utils.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))
    has_nan = mrna_GC.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # 9. K-mers (All now return fixed-size lists)
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

    # 10. siRNA rules codes (PADDED to 19*3)
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data['siRNA_seq']:
        sirna_pos_scores.append(utils.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    has_nan = sirna_pos_scores.isnull().values.any()
    if has_nan:
        raise ValueError("Failed feature processing")

    # ## mRNA-AGO2
    # mrna_ago = pd.read_csv(f"src/data/input/RNA_AGO2/mRNA_AGO2_zh.csv",index_col=0)
    # mrna_ago = mrna_ago.reindex(mrna_onehot.index)
    # has_nan = mrna_ago.isnull().values.any()
    # if has_nan:
    #     raise ValueError

    # sirna_ago = pd.read_csv(f"src/data/input/RNA_AGO2/siRNA_AGO2_zh.csv",index_col = 0)
    # sirna_ago = sirna_ago.reindex(sirna_onehot.index)
    # has_nan = sirna_ago.isnull().values.any()
    # if has_nan:
    #     raise ValueError
    
    '''

    The features of GNN nodes

    '''

    # The features of GNN nodes
    sirna_pd  = pd.concat([sirna_ago, sirna_k_mers, sirna_GC, sirna_pos_scores, sirna_onehot,sirna_sfold_feat], axis=1)

    mrna_pd = pd.concat([mrna_sfold_feat,  mrna_ago,  mrna_GC, mrna_onehot], axis=1)

    input_dim_mRNA = mrna_pd.shape[1]
    input_dim_siRNA = sirna_pd.shape[1]
    input_dim_thermo = sirna_thermo_feat.shape[1]     

    dataset = EfficacyDataset(data, sirna_pd, mrna_pd, sirna_thermo_feat)
    data_loader = DataLoader(dataset, params["batch_size"], shuffle = False, collate_fn = collate_fn)
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EfficacyModel(input_dim_mRNA, input_dim_siRNA, input_dim_thermo).cuda()

    # 2. Load the saved weights
    path=f"src/checkpoints/best_model_fold.pt"
    try:
        model.load_state_dict(torch.load(path))
    except Exception as e:
        raise FileNotFoundError(f"Model loading failed: {e}")

    # 3. Set the model to evaluation mode (if evaluating or testing)
    model.eval()
    
    @torch.no_grad()
    def inference(model, dataloader, device):
        model.eval()
        all_preds = []
    
        with torch.no_grad():
            for batch in dataloader:
                # NEW: Unpack includes siRNA sequence  
                mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq = [b.to(device) for b in batch]
    
                # Forward pass with sequence
                y_pred, _ = model(mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq)
    
                all_preds.append(y_pred.cpu())
    
        # Rest of validation metrics calculation stays the same...
        all_preds = torch.cat(all_preds)

        return all_preds
        
    try:    
        inference_preds = inference(model, data_loader, device)  # inference_loader = your DataLoader for inference dataset

        inference_preds = inference_preds.numpy().flatten()

        print("Predictions:", inference_preds[0])
            
        return inference_preds[0]
    
    except Exception as e:
        
        raise RuntimeError(f"Inference failed: {e}")
    
if __name__ == "__main__":
    main()
    


