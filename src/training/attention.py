import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
# Import the revised utils1.py
import BugUtils as utils1

# --- Load parameters ---
with open("siRNA_param_pytorch.json", 'r') as f:
    params = json.load(f)

# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]

# Initialize metric lists
score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []
score_f1=[]
score_recall = []
score_prec = []

# Initialize test metric lists
test_score_PCC = []
test_score_SPCC = []
test_score_mse = []
test_score_auc = []
test_score_f1 = []
test_score_recall = []
test_score_prec = []

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
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
        self.nucleotide_embedding = nn.Embedding(5, embed_dim)  # A, T, G, C/U -> embed_dim
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
        mRNA_slice_embedded = self.sequence_proj(self.nucleotide_embedding(mRNA_slice))  # (B,57,embed_dim)

        # Cross-attention: query = mRNA slice, key/value = siRNA sequence
        cross_attended1 = self.cross_attention(
            query=mRNA_slice_embedded,   # (B,57,embed_dim)
            key_value=siRNA_seq_embedded # (B,Ls,embed_dim)
        )  # (B,57,embed_dim)

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
    # def forward(self, mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_slice):
    #     print("\n---- Forward Pass ----")
    #     print("mRNA_x:", mRNA_x.shape)
    #     print("siRNA_x:", siRNA_x.shape)
    #     print("thermo_x:", thermo_x.shape)
    #     print("siRNA_seq:", siRNA_seq.shape)
    #     print("mRNA_slice:", mRNA_slice.shape)
    
    #     mRNA_proj = self.ln_proj(self.proj_mRNA(mRNA_x))
    #     siRNA_proj = self.ln_proj(self.proj_siRNA(siRNA_x))
    #     print("mRNA_proj:", mRNA_proj.shape)
    #     print("siRNA_proj:", siRNA_proj.shape)
    
    #     mRNA_hidden = self.shared_encoder(mRNA_proj)
    #     siRNA_hidden = self.shared_encoder(siRNA_proj)
    #     print("mRNA_hidden:", mRNA_hidden.shape)
    #     print("siRNA_hidden:", siRNA_hidden.shape)
    
    #     # Embeddings
    #     siRNA_seq_embedded = self.sequence_proj(self.nucleotide_embedding(siRNA_seq))
    #     mRNA_slice_embedded = self.sequence_proj(self.nucleotide_embedding(mRNA_slice))
    #     print("siRNA_seq_embedded:", siRNA_seq_embedded.shape)
    #     print("mRNA_slice_embedded:", mRNA_slice_embedded.shape)
    
    #     # Cross-attention outputs
    #     cross_attended1 = self.cross_attention(mRNA_slice_embedded, siRNA_seq_embedded)
    #     cross_attended2 = self.cross_attention(siRNA_seq_embedded, mRNA_slice_embedded)
    #     print("cross_attended1:", cross_attended1.shape)
    #     print("cross_attended2:", cross_attended2.shape)
    
    #     # Pooling
    #     cross_attended1 = cross_attended1.mean(dim=1)
    #     cross_attended2 = cross_attended2.mean(dim=1)
    #     cross_attended = torch.stack([cross_attended1, cross_attended2], dim=0).mean(dim=0)
    #     print("cross_attended (pooled):", cross_attended.shape)
    
    #     fusion = torch.cat([mRNA_hidden, siRNA_hidden], dim=-1)
    #     print("fusion:", fusion.shape)
    
    #     h = self.ln_fusion(F.relu(self.fc_fusion(fusion)))
    #     print("h:", h.shape)
    
    #     h_thermo = torch.cat([h, cross_attended, thermo_x], dim=-1)
    #     print("h_thermo:", h_thermo.shape)

    #     h_mlp = self.ln_mlp1(F.relu(self.fc_mlp1(h_thermo)))
    #     print("h_mlp:", h_mlp.shape)
    
    #     out = self.fc_out(h_mlp)
    #     print("out:", out.shape)
    #     print("---- Forward End ----\n")
    
    #     return out.squeeze(-1), h



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

        # # NEW: mRNA slice (100nt around pos)
        # mrna_seq = row["mRNA_seq"]
        # pos = int(row["pos"])
        # start = max(0, pos - 50)
        # end = min(len(mrna_seq), pos + 50)
        # mrna_slice = mrna_seq[start:end]
        
        # # pad if <100
        # if len(mrna_slice) < 100:
        #     mrna_slice = mrna_slice + "N" * (100 - len(mrna_slice))


        mrna_slice_tensor = self.sequence_to_tensor(mrna_slice)

        label = torch.tensor(row["efficacy"], dtype=torch.float32)
        return mrna_x, sirna_x, thermo_x, sirna_seq_tensor, mrna_slice_tensor, label

    def sequence_to_tensor(self, sequence):
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'U': 1, 'N': 4}
        return torch.tensor([nucleotide_map.get(nt.upper(), 0) for nt in sequence], dtype=torch.long)



def collate_fn(batch):
    mrna_list, sirna_list, thermo_list, sirna_seq_list, mrna_slice_list, labels = zip(*batch)

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

# --- K-Fold Cross Validation Loop ---
NUM_FOLDS = 10
for n in range(NUM_FOLDS):
    print(f"\nProcessing fold {n}")

    # Read and preprocess data
    split_dir = f"siRNA_split_datasets/split{n}"
    if not os.path.exists(split_dir):
        print(f"Error: Directory '{split_dir}' not found. Please ensure your data splits are correctly organized.")
        print("Skipping fold processing.")
        continue

    try:
        data_train = pd.read_csv(os.path.join(split_dir, "train.csv"))
        data_dev = pd.read_csv(os.path.join(split_dir, "dev.csv"))
        data_test = pd.read_csv(os.path.join(split_dir, "test.csv"))
    except FileNotFoundError as e:
        print(f"Error: Data file not found in '{split_dir}'. {e}")
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

    print("\n--- Feature Processing ---")

    # --- Feature processing with variable length handling and padding ---

    # 1. One-hot encoding for siRNA (PADDED to MAX_SIRNA_LENGTH)
    sirna_onehot = []
    for seq in data['siRNA_seq']:
        sirna_onehot.append(utils1.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data['siRNA']))

    has_nan = sirna_onehot.isnull().values.any()
    print(f"siRNA onehot Contains NaN: {has_nan}")

    # 2. mRNA one-hot (PADDED to max_mrna_len)
    mrna_onehot_temp = data.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
    mrna_onehot = [utils1.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
                   for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))
    has_nan = mrna_onehot.isnull().values.any()
    print(f"mRNA onehot Contains NaN: {has_nan}")

    # # --- Sequence Embedding using MP-RNA Transformer ---
    # print("--- Starting Sequence Embedding (siRNA) ---")
    # sirna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in data['siRNA_seq']]
    # sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(data['siRNA']))
    # sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')]
    # print("--- Finished Sequence Embedding (siRNA) ---")

    # print("--- Starting Sequence Embedding (mRNA) ---")
    # mrna_unique_seq_df = data.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA")
    # mrna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    # mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index = list(mrna_unique_seq_df['mRNA']))
    # mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')]
    # print("--- Finished Sequence Embedding (mRNA) ---")

    # 3. Positional encoding (PADDED to MAX_SIRNA_LENGTH * dmodel)
    sirna_pos_encoding = []
    for idx, row in data.iterrows():
        mrna_start_pos = max(0, int(row['pos']))
        sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
            mrna_start_pos,
            len(row['siRNA_seq']),
            MAX_SIRNA_LENGTH,
            params["dmodel"]
        ))
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data['siRNA'] + '_' + data['mRNA'])
    has_nan = sirna_pos_encoding.isnull().values.any()
    print(f"siRNA positional Contains NaN: {has_nan}")

    # 4. Thermodynamics (PADDED to MAX_SIRNA_LENGTH)
    sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                         for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index')

    # 5. Co-fold features (using base_pair_probs as indicated by user)
    con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt", header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)
    has_nan = con_feat.isnull().values.any()
    print(f"con features Contains NaN: {has_nan}")

    # 6. Self-fold features (siRNA)
    sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)
    has_nan = sirna_sfold_feat.isnull().values.any()
    print(f"sirna_sfold_feat Contains NaN: {has_nan}")

    # 7. Self-fold features (mRNA)
    mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    has_nan = mrna_sfold_feat.isnull().values.any()
    print(f"mrna_sfold_feat Contains NaN: {has_nan}")

    # AGO2
    ## siRNA-AGO2
    sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2_zh.csv",index_col = 0)
    sirna_ago = sirna_ago.reindex(sirna_onehot.index)
    has_nan = sirna_ago.isnull().values.any()
    print(f"sirna_ago Contains NaN: {has_nan}")

    ## mRNA-AGO2
    mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2_zh.csv",index_col=0)
    mrna_ago = mrna_ago.reindex(mrna_onehot.index)
    has_nan = mrna_ago.isnull().values.any()
    print(f"mrna_ago Contains NaN: {has_nan}")

    # 8. GC percentage (variable length robust)
    sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data['siRNA_seq']], index=list(data['siRNA']))
    has_nan = sirna_GC.isnull().values.any()
    print(f"sirna_GC Contains NaN: {has_nan}")
    mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))
    has_nan = mrna_GC.isnull().values.any()
    print(f"mrna_GC Contains NaN: {has_nan}")

    # 9. K-mers (All now return fixed-size lists)
    sirna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data['siRNA']
    has_nan = sirna_k_mers.isnull().values.any()
    print(f"sirna_k_mers Contains NaN: {has_nan}")

    # 10. siRNA rules codes (PADDED to 19*3)
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data['siRNA_seq']:
        sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    has_nan = sirna_pos_scores.isnull().values.any()
    print(f"sirna_pos_scores Contains NaN: {has_nan}")

    print("\n--- Assembling GNN Node Features ---")

    # siRNA nodes features
    # sirna_pd  = pd.concat([sirna_ago, sirna_k_mers, sirna_GC, sirna_pos_scores, sirna_onehot, sirna_embedding_df], axis=1)
    sirna_pd  = pd.concat([sirna_ago, sirna_k_mers, sirna_GC, sirna_pos_scores, sirna_onehot,sirna_sfold_feat], axis=1)
    #sirna_pd  = pd.concat([sirna_ago, sirna_k_mers, sirna_GC, sirna_pos_scores, sirna_onehot], axis=1)

    # mRNA nodes features
    # mrna_pd = pd.concat([mrna_embedding_df,  mrna_ago,  mrna_GC,  mrna_onehot], axis=1)
    mrna_pd = pd.concat([mrna_sfold_feat,  mrna_ago,  mrna_GC, mrna_onehot], axis=1)
    #mrna_pd = pd.concat([ mrna_ago,  mrna_GC, mrna_onehot], axis=1)

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
    model = EfficacyModel(input_dim_mRNA, input_dim_siRNA, input_dim_thermo).cuda()
    #model = EfficacyModel(input_dim_mRNA, input_dim_siRNA,).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ✅ Print total trainable parameters for this fold
    # ✅ Print total trainable parameters for this fold
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"\n✅ Total Trainable Parameters in Fold {n}: {total_params:,}\n")




    # def train(model, dataloader, optimizer, device,use_contrastive=False, lambda_metric=0.1):
    #     model.train()
    #     total_loss = 0

    #     for batch in dataloader:
    #         mRNA_x, siRNA_x, y_true,mrna_mask, sirna_mask = [b.to(device) for b in batch]

    #         # print(siRNA_x)
    #         # print(mRNA_x)

    #         # Forward pass
    #         y_pred, metric_emb = model(mRNA_x, siRNA_x)


    #         # print(y_pred)
    #         # print(y_true)
    #         # Regression loss
    #         reg_loss = F.mse_loss(y_pred, y_true)

    #         # Optional metric learning
    #         # if use_contrastive:
    #         # Here, label = 1 if y_true > threshold (effective), else 0
    #         label = (y_true > 0.7).float()
    #         emb_m, emb_s = metric_emb.chunk(2, dim=-1)
    #         metric_loss = contrastive_loss(emb_m, emb_s, label)
    #         loss = reg_loss + lambda_metric * metric_loss
    #         # else:
    #         #     loss = reg_loss

    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()

    #         total_loss += loss.item()

    #     return total_loss / len(dataloader)

    def train(model, dataloader, optimizer, device, use_contrastive=True, lambda_metric=0.1):
        model.train()
        total_loss = 0
    
        for batch in dataloader:
            # NEW: Unpack includes siRNA sequence
            mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq, y_true = [b.to(device) for b in batch]
            # print(siRNA_seq)
    
            # Forward pass with sequence
            y_pred, metric_emb = model(mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq)
            # Rest stays the same...
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

    def validate(model, dataloader, device):
        model.eval()
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for batch in dataloader:
                # NEW: Unpack includes siRNA sequence  
                mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq, y_true = [b.to(device) for b in batch]
    
                # Forward pass with sequence
                y_pred, _ = model(mRNA_x, siRNA_x, thermo_x, siRNA_seq, mRNA_seq)
    
                all_preds.append(y_pred.cpu())
                all_labels.append(y_true.cpu())
    
        # Rest of validation metrics calculation stays the same...
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Calculate metrics (same as before)
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
        
    # #print(f"--- Fold {n} finished! Final Validation Loss: {best_val_loss:.4f} ---")

    # Evaluate on test set after training is complete
    print(f"\n--- Testing on fold {n} ---")
    eval_model = EfficacyModel(input_dim_mRNA, input_dim_siRNA, input_dim_thermo).cuda()
    # Load the saved weights
    model_path = f"attn_model_fold{n}.pt" # <--- IMPORTANT: Ensure this path is correct for your saved models
    # model_path = f"best_model_fold1_gcn_75.pt"
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
print("Overall MSE score =", np.mean(test_score_mse))
print("Overall PCC score =", np.mean(test_score_PCC))
print("Overall SPCC score =", np.mean(test_score_SPCC))
print("Overall AUC score =", np.mean(test_score_auc))
print("Overall F1 score =", np.mean(test_score_f1))
print("Overall Precision score =", np.mean(test_score_prec))
print("Overall Recall score =", np.mean(test_score_recall))
print("Model checkpoints saved based on best validation loss for each fold.")