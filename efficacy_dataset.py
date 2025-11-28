import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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

