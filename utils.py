import numpy
import math
import itertools 
import torch


position_scores = [
    {"A": -1, "C": 1, "G": 1, "U": -1},
    {"A": -1, "C": 0, "G": 1, "U": -1},
    {"A": 1, "C": -1, "G": 1, "U": -1},
    {"A": 0, "C": -1, "G": 0, "U": 1},
    {"A": 1, "C": 0, "G": 0, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": 1, "U": -1},
    {"A": 1, "C": 0, "G": -1, "U": 0},
    {"A": 0, "C": 0, "G": -1, "U": -1},
    {"A": 1, "C": 1, "G": 1, "U": 1},
    {"A": 0, "C": 1, "G": 1, "U": 0},
    {"A": 1, "C": 0, "G": -1, "U": 0},
    {"A": 1, "C": -1, 'G': -1, "U": 1},
    {"A": 0, "C": -1, "G": 0, "U": 0},
    {"A": 1, "C": -1, "G": 0, "U": -1},
    {"A": 0, "C": 0, "G": 1, "U": 1},
    {"A": 1, "C": 0, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1}
]



# Transfering seq to one-hot
def obtain_one_hot_feature_for_one_sequence_1(seq1, max_len):
    """
    Converts a sequence to one-hot encoding, padding or truncating to max_len.
    """
    mapping = dict(zip("NACGT", range(5)))
    
    seq_numeric = [mapping.get(i.upper(), 0) for i in seq1]

    if len(seq_numeric) > max_len:
        seq_numeric = seq_numeric[:max_len]
    
    padding_len = max_len - len(seq_numeric)
    
    unit_arr = numpy.concatenate((numpy.zeros((1, 4), dtype=numpy.uint8), numpy.eye(4, dtype=numpy.uint8)))

    encoded_seq = unit_arr[seq_numeric]
    
    zero_arr = numpy.zeros((padding_len, 4), dtype=numpy.uint8)
    
    return numpy.concatenate((encoded_seq, zero_arr)).flatten()




def get_pos_embedding_sequence(mrna_start_pos, len_sirna, max_sirna_len, d_model):
    """
    Generates positional embeddings for an siRNA sequence, padding to max_sirna_len.
    """
    pe_list = []
    for offset in range(len_sirna):
        t = mrna_start_pos + offset
        for i in range(d_model):
            pe = get_pos_embedding(i, d_model, t)
            pe = round(pe, 4)
            pe_list.append(pe)
    
    expected_padded_len = max_sirna_len * d_model
    
    if len(pe_list) < expected_padded_len:
        pe_list.extend([0.0] * (expected_padded_len - len(pe_list)))
    elif len(pe_list) > expected_padded_len:
        pe_list = pe_list[:expected_padded_len]
        
    return pe_list


# Thermodynamics (padding zeros will remain as it's for fixed output length)
def build_kmers(sequence):
    """Builds 2-mers (dinucleotides) from a sequence."""
    kmers = []
    if len(sequence) >= 2: # This guard is for `len(sequence) - 1` in the loop
        n_kmers = len(sequence) - 1
        for i in range(n_kmers):
            kmer = sequence[i:i + 2]
            kmers.append(kmer)
    return kmers

def cal_thermo_feature(sequence, max_sirna_len, intermolecular_initiation=4.09, symmetry_correction=0.43):
    """
    Calculates thermodynamic features for an siRNA sequence, padding to max_sirna_len.
    """
    seq = sequence.upper().replace("T", "U")
    
    sum_stability = 0
    single_sum_values = []

    if len(seq) > 0 and seq[0] == 'A':
        sum_stability += 0.45
    if len(seq) >= 19 and seq[18] == 'U':
        sum_stability += 0.45

    bimers = build_kmers(seq)

    bimer_values_dict = {
        'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33,
        'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11,
        'GU': -2.24, 'AC': -2.24, 'GA': -2.35, 'UC': -2.35,
        'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42
    }

    for b in bimers:
        stability_value = bimer_values_dict.get(b, 0)
        single_sum_values.append(stability_value)
        sum_stability += stability_value

    sum_stability += intermolecular_initiation
    sum_stability += symmetry_correction
    
    single_sum_values.append(round(sum_stability, 2))

    expected_output_len = max_sirna_len 
    
    if len(single_sum_values) < expected_output_len:
        single_sum_values.extend([0.0] * (expected_output_len - len(single_sum_values)))
    elif len(single_sum_values) > expected_output_len:
        single_sum_values = single_sum_values[:expected_output_len]
    
    return single_sum_values

# count GC percentage 
def countGC(seq):
    seq = seq.upper()
    gc_count = (seq.count("G") + seq.count("C"))
    gc_percent = gc_count / len(seq)
    return round(gc_percent, 3)


# K-mer functions
def get_kmer_freq(seq, k):
    """
    Calculates k-mer frequencies. Returns a fixed-size list of frequencies for all possible k-mers.
    WARNING: Will raise an error if the sequence is too short to form k-mers (len(seq) < k).
    """
    seq = seq.upper().replace("T","U")
    
    bases = "ACGU"
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    
    kmer_freq = {kmer: 0.0 for kmer in all_kmers}
    
    total_kmers = len(seq) - k + 1
    for i in range(total_kmers):
        kmer = seq[i:i+k]
        if kmer in kmer_freq:
            kmer_freq[kmer] += 1
    
    # This division will raise an error if total_kmers is 0 or negative
    for key in kmer_freq:
        kmer_freq[key] /= total_kmers
        
    return list(kmer_freq.values())

# Wrappers for specific k-mer lengths
def single_freq(seq, *args):
    return get_kmer_freq(seq, 1)

def double_freq(seq, *args):
    return get_kmer_freq(seq, 2)

def triple_freq(seq, *args):
    return get_kmer_freq(seq, 3)

def quadruple_freq(seq, *args):
    return get_kmer_freq(seq, 4)

def quintuple_freq(seq, *args):
    return get_kmer_freq(seq, 5)


# Calculate siRNA rules codes (padding zeros will remain for fixed output length)
def one_hot_encode(score):
    if score == -1:
        return [1, 0, 0]
    elif score == 0:
        return [0, 1, 0]
    elif score == 1:
        return [0, 0, 1]
    return [0, 0, 0]


def rules_scores(seq, max_rules_length=19):
    """
    Obtain siRNA each position scores with one-hot encoding, padded to a fixed length.
    """
    seq = seq.upper().replace("T","U")
    one_hot_scores = []
    
    for i in range(max_rules_length):
        if i < len(seq) and i < len(position_scores):
            score = position_scores[i].get(seq[i], 0)
            one_hot_scores.extend(one_hot_encode(score))
        else:
            one_hot_scores.extend([0, 0, 0])
    
    expected_len = max_rules_length * 3
    if len(one_hot_scores) < expected_len:
        one_hot_scores.extend([0] * (expected_len - len(one_hot_scores)))
    elif len(one_hot_scores) > expected_len:
        one_hot_scores = one_hot_scores[:expected_len]

    return one_hot_scores

