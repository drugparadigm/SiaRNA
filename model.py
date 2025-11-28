import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key_value, mask=None):
        B, q_len, _ = query.size()
        kv_len = key_value.size(1)

        Q = self.query_proj(query)
        K = self.key_proj(key_value)
        V = self.value_proj(key_value)

        Q = Q.view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, V)

        attended = attended.transpose(1, 2).contiguous().view(B, q_len, self.embed_dim)
        return self.out_proj(attended)

class SiaRNA(nn.Module):
    def __init__(self, input_dim_mRNA, input_dim_siRNA, input_dim_thermo,
                 proj_dim=256, hidden_dim=64, embed_dim=32, fusion_dim=128, mlp_dim=64):
        super().__init__()

        self.proj_mRNA = nn.Linear(input_dim_mRNA, proj_dim)
        self.proj_siRNA = nn.Linear(input_dim_siRNA, proj_dim)
        self.ln_proj = nn.LayerNorm(proj_dim)
        self.shared_encoder = SequenceEncoder(proj_dim, hidden_dim, embed_dim)

        self.nucleotide_embedding = nn.Embedding(5, embed_dim)
        self.sequence_proj = nn.Linear(embed_dim, embed_dim)

        self.cross_attention = CrossAttention(embed_dim, num_heads=4)

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

        siRNA_seq_emb = self.sequence_proj(self.nucleotide_embedding(siRNA_seq))
        mRNA_seq_emb = self.sequence_proj(self.nucleotide_embedding(mRNA_slice))

        cross1 = self.cross_attention(query=mRNA_seq_emb, key_value=siRNA_seq_emb)
        cross2 = self.cross_attention(query=siRNA_seq_emb, key_value=mRNA_seq_emb)

        cross1 = cross1.mean(dim=1)
        cross2 = cross2.mean(dim=1)

        cross_attended = torch.stack([cross1, cross2], dim=0).mean(dim=0)

        fusion = torch.cat([mRNA_hidden, siRNA_hidden], dim=-1)
        h = self.ln_fusion(F.relu(self.fc_fusion(fusion)))

        h = torch.cat([h, cross_attended, thermo_x], dim=-1)
        h = self.ln_mlp1(F.relu(self.fc_mlp1(h)))
        return self.fc_out(h).squeeze(-1), h