#preprocessing
from Bio import SeqIO
import pandas as pd
import subprocess
import itertools
import os
import glob
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import save_npz
import re
import torch
import sys
from flask import request


def main(request_Id: str, mRNA_path: str, siRNA_path:str):
    current_dir="data/input/"
    # print("Preprocessing-----------")
        
    reqId = request_Id
    main_dir = os.getcwd()
    # os.chdir('src')
    # Read mrna file
    # mrna_file = f"data/input/{reqId}_mRNA.fa"
    mrna_file=mRNA_path
    mrna_map = []
    allowed_chars_mrna = {'A', 'T', 'C', 'G', 'N'}
    for seq_record in SeqIO.parse(mrna_file,"fasta"):
        mrna_id = seq_record.id
        seq = str(seq_record.seq).upper().strip()
        if len(seq)>2000 or len(seq)<21:
            raise ValueError(f"Invalid length of mRNA sequence {len(seq)}")
        if any(base not in allowed_chars_mrna for base in seq):
            raise ValueError(f"mRNA sequence contains un-recognized characters")
        mrna_map.append((mrna_id, str(seq)))
    
    mrna_df = pd.DataFrame(mrna_map, columns=['mRNA', 'mRNA_seq'])
    
    # Read sirna file
    # sirna_file = f"data/input/{reqId}_siRNA.fa"  # assuming FASTA format for siRNAs
    sirna_file=siRNA_path
    sirna_map = []
    allowed_chars_sirna = {'A', 'U', 'C', 'G', 'N'}
    for seq_record in SeqIO.parse(sirna_file, "fasta"):
        sirna_id = seq_record.id
        seq = str(seq_record.seq).upper().strip()
        if len(seq) != 21:
            raise ValueError(f"Invalid length of siRNA sequence {len(seq)}")
        if any(base not in allowed_chars_sirna for base in seq):
            raise ValueError(f"siRNA sequence contains un-recognized characters")
        sirna_map.append((sirna_id, seq))
    sirna_df = pd.DataFrame(sirna_map, columns=['siRNA', 'siRNA_seq'])
    
    # Create **all possible siRNA–mRNA pairs** for inference
    pairs_list = list(itertools.product(sirna_df.itertuples(index=False),
                                        mrna_df.itertuples(index=False)))
    
    # Prepare pairs DataFrame
    pair_data = []
    for sirna_row, mrna_row in pairs_list:
        pair_data.append({
            'siRNA': sirna_row.siRNA,
            'siRNA_seq': sirna_row.siRNA_seq,
            'mRNA': mrna_row.mRNA,
            'mRNA_seq': mrna_row.mRNA_seq,
            'seq_pairs': sirna_row.siRNA_seq + '&' + mrna_row.mRNA_seq
        })
    
    pairs_df = pd.DataFrame(pair_data)
    
    
    path = "data/input/"
    files = os.listdir(path)
    
    current_path = os.getcwd()
    output_path_dir = os.path.join(current_path, f"data/input/{reqId}_preprocess")
            
    os.makedirs(output_path_dir,exist_ok=True)
                
    
    #---------------------mRNA--------------------
    # RNAcofold
    for ind, row in mrna_df.iterrows():
    	seq = row[1]
    	
    	id_name = reqId+"_"+row[0]
    
    
    	proc = subprocess.Popen(['RNAfold','-p',"--id-prefix=" + id_name], 
    		stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True,cwd="data/input")
    
    	output, error = proc.communicate(seq)
    	
    path_mrna = "data/input/"
    files_mrna = os.listdir(path_mrna)
    for file in files_mrna:
        if "_0001_dp.ps" not in file or '.bpp' in file:
            continue
        name = file.replace("_0001_dp.ps", "")
        temp = open(path_mrna + file).readlines()
        start_flag = False
        os.makedirs(f"data/input/{reqId}_RNAfold_bp_file_mrna", exist_ok=True)
        f_mrna = open(f"data/input/{reqId}_RNAfold_bp_file_mrna/" + file + ".bpp", "w")
    
        for line in temp:
            line = line.strip()
            if "start of base pair probability data" in line:
                start_flag = True
            if start_flag == True and "ubox" in line:
                line = line.strip().split()
                assert(len(line) == 4)
                i, j, prob, _ = line
                prob = float(prob)
                f_mrna.write(str(i) + " " + str(j) + " " + str(prob*prob) + "\n")
        f_mrna.close()
            
    #---------------3---------
    
    matrix_size_mrna = 9756
    n_components = 100
    
    file_paths_mrna = glob.glob(f'data/input/{reqId}_RNAfold_bp_file_mrna/*dp.ps.bpp')
    
    for file_path in file_paths_mrna:
        pos_data = np.loadtxt(file_path, usecols=[0, 1, 2])
        pos_data_gpu = np.asarray(pos_data) 
    
    
        pos_matrix = np.zeros((matrix_size_mrna, matrix_size_mrna), dtype=np.float32) 
        pos_matrix[pos_data_gpu[:, 0].astype(np.int32) - 1, pos_data_gpu[:, 1].astype(np.int32) - 1] = pos_data_gpu[:, 2]
        pos_matrix = pos_matrix + pos_matrix.T - np.diag(np.diag(pos_matrix))
          
          
        data_matrix_cpu = pos_matrix
        sparse_matrix = csr_matrix(data_matrix_cpu)
    
            
        svd = TruncatedSVD(n_components=100, random_state=0)
        reduced_data = svd.fit_transform(sparse_matrix)
        directory_path_mrna = f'data/input/{reqId}_RNAfold_reduced_matrix_mrna'+ str(n_components)
    
        if not os.path.exists(directory_path_mrna):
            os.mkdir(directory_path_mrna)
    
        np.save(f'data/input/{reqId}_RNAfold_reduced_matrix_mrna'+ str(n_components) + '/' + file_path.split('/')[-1].split('_dp.ps.bpp')[0]+'.npy',reduced_data)
            
    #------------------4---------------------
    
    path_mrna = f"data/input/{reqId}_RNAfold_reduced_matrix_mrna100"
    files = os.listdir(path_mrna)
    
    
    df_mrna = []
    first_parts_mrna = []
    
    for file in files:
        file_name = file.replace('_0001.npy', '')
        file_name=file_name.replace(f'{reqId}_', '')
        parts = file_name.split('_0001', 1)
        first_parts_mrna.append(parts[0])
    
        data = np.load(path_mrna + "/" + file)
        data = data.mean(0)
        df_mrna.append(data)
    
    
    
    df_mrna = [np.random.randn(100).astype(np.float32) for _ in range(2)]
    
    # create the DataFrame
    df_res_mrna = pd.DataFrame(df_mrna)
    
    first_parts_mrna = first_parts_mrna * len(df_res_mrna)
    
    df_res_mrna.index = first_parts_mrna
    
    output_path_mrna = os.path.join(output_path_dir,f"{reqId}_con_matrix_meanSum100.txt")
    df_res_mrna.iloc[[0]].to_csv(output_path_mrna, header=False)
    
    
    
    # Loop over all files in the current directory
    for filename in os.listdir(current_dir):
        if filename.startswith(reqId) and filename.endswith(".ps"):
            file_path = os.path.join(current_dir, filename)
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
    #---------------------siRNA-------------------
    # RNAcofold
    for ind, row in sirna_df.iterrows():
    	seq = row[1]
    	
    	id_name = reqId+'_'+row[0]
    	
    	proc = subprocess.Popen(['RNAfold','-p',"--id-prefix=" + id_name], 
    		stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True,cwd="data/input")
    
    	output, error = proc.communicate(seq)
    	
    path_sirna = "data/input/"
    files_sirna = os.listdir(path_sirna)
    
    #-------------2------------
    
    for file in files_sirna:
        if "_0001_dp.ps" not in file or '.bpp' in file:
            continue
        name = file.replace("_0001_dp.ps", "")
        temp = open(path_sirna + file).readlines()
        start_flag = False
        os.makedirs(f"data/input/{reqId}_RNAfold_bp_file_sirna", exist_ok=True)
        f_sirna = open(f"data/input/{reqId}_RNAfold_bp_file_sirna/" + file + ".bpp", "w")
        for line in temp:
            line = line.strip()
            if "start of base pair probability data" in line:
                start_flag = True
            if start_flag == True and "ubox" in line:
                line = line.strip().split()
                assert(len(line) == 4)
                i, j, prob, _ = line
                prob = float(prob)
                f_sirna.write(str(i) + " " + str(j) + " " + str(prob*prob) + "\n")
        f_sirna.close()
            
    #-------------3-----------------
    matrix_size_sirna = 21
    n_components_sirna = 6  
    
    file_paths_sirna = glob.glob(f'data/input/{reqId}_RNAfold_bp_file_sirna/*dp.ps.bpp')
    
    for file_path in file_paths_sirna:
        try:
            pos_data = np.loadtxt(file_path, usecols=[0, 1, 2])  
            if len(pos_data) < n_components_sirna: 
    
                reduced_data = np.zeros((matrix_size_sirna, 6))
            else:
                pos_data_gpu = np.asarray(pos_data)  
                pos_matrix = np.zeros((matrix_size_sirna, matrix_size_sirna), dtype=np.float32)  
                pos_matrix[pos_data_gpu[:, 0].astype(np.int32) - 1, pos_data_gpu[:, 1].astype(np.int32) - 1] = pos_data_gpu[:, 2] 
                pos_matrix = pos_matrix + pos_matrix.T - np.diag(np.diag(pos_matrix))
    
                data_matrix_cpu = pos_matrix
                sparse_matrix = csr_matrix(data_matrix_cpu)
    
                svd = TruncatedSVD(n_components=6, random_state=0)
                reduced_data = svd.fit_transform(sparse_matrix)
        except:
            reduced_data = np.zeros((matrix_size_sirna, n_components_sirna))
     
        
        directory_path_sirna = f'data/input/{reqId}_RNAfold_reduced_matrix_sirna'+ str(n_components_sirna)
    
        if not os.path.exists(directory_path_sirna):
            os.mkdir(directory_path_sirna)
    
        np.save(f'data/input/{reqId}_RNAfold_reduced_matrix_sirna'+ str(n_components_sirna) + '/' + file_path.split('/')[-1].split('_dp.ps.bpp')[0]+'.npy',reduced_data)
                
    #------------------4--------------
    path_sirna = f"data/input/{reqId}_RNAfold_reduced_matrix_sirna6"
    files_sirna = os.listdir(path_sirna)
    
    df_sirna = []
    first_parts_sirna = []
    
    for file in files_sirna:
        file_name = file.replace('_0001.npy', '')
        file_name = file_name.replace(f'{reqId}_', '')
        parts = file_name.split('_0001', 1)
        first_parts_sirna.append(parts[0])
    
        data = np.load(os.path.join(path_sirna, file))
        result_data = data.mean(0)
    
        df_sirna.append(result_data)
    
    
    # create the DataFrame
    df_res_sirna = pd.DataFrame(df_sirna)
    first_parts_sirna = first_parts_sirna * len(df_res_sirna)
    
    df_res_sirna.index = first_parts_sirna
    
    output_path_sirna = os.path.join(output_path_dir,f"{reqId}_self_siRNA_matrix_meanSum6.txt")
    df_res_sirna.iloc[[0]].to_csv(output_path_sirna, header=False)
    
    
    
    # Loop over all files in the current directory
    for filename in os.listdir(current_dir):
        if filename.startswith(reqId) and filename.endswith(".ps"):
            file_path = os.path.join(current_dir, filename)
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
    #---------------------match position------------------------------
    
    # --- Load siRNA sequences ---
    sirna_map = {
        rec.id: str(rec.seq).upper().replace("T", "U")
        for rec in SeqIO.parse(siRNA_path, "fasta")
    }
    
    # --- Load mRNA sequences ---
    mrna_map = {
        rec.id: str(rec.seq).upper().replace("T", "U")
        for rec in SeqIO.parse(mRNA_path, "fasta")
    }
    
    # --- Generate all possible siRNA-mRNA pairs ---
    pairs = []
    for sirna_id, sirna_seq in sirna_map.items():
        for mrna_id, mrna_seq in mrna_map.items():
            pairs.append({
                'siRNA': sirna_id,
                'mRNA': mrna_id,
                'siRNA_seq': sirna_seq,
                'mRNA_seq_RNA-FM': mrna_seq,
                'mRNA_seq': mrna_seq.replace("U", "T")  # For other analysis
            })
    
    # Convert to DataFrame
    eff_df = pd.DataFrame(pairs)
    
    # --- Compute match position (optional) ---
    def reverse_complement_rna(seq):
        return seq.translate(str.maketrans("AUGC", "UACG"))[::-1]
    
    def find_match_pos(row):
        try:
            match = reverse_complement_rna(row['siRNA_seq'])
            return row['mRNA_seq_RNA-FM'].find(match)
        except:
            raise ValueError(f"Match postion not found")
    
    eff_df['pos'] = eff_df.apply(find_match_pos, axis=1)
    
    # (Optional) Remove pairs with no reverse complement match
    eff_df = eff_df[eff_df['pos'] != -1]
    
    # --- Save final merged dataset ---
    final_cols = ['siRNA', 'mRNA', 'siRNA_seq', 'mRNA_seq', 'mRNA_seq_RNA-FM', 'pos']
    # print("Creating file")
    eff_df[final_cols].to_csv(f"data/input/{reqId}_inference_siRNA_mRNA_pairs.csv", index=False)
    
    # print("Inference preprocessing complete. Output saved to 'inference_siRNA_mRNA_pairs.csv'")
    # os.chdir(main_dir)

