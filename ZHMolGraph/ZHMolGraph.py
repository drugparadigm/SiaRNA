import os
from ZHMolGraph.import_modules import *
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from tqdm import tqdm
import json
import time
import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import sys
import argparse
import pyhocon
import random
import csv
import fm
import warnings
warnings.filterwarnings("ignore")

from graphsage_src.My_dataCenter import *
from graphsage_src.utils import *
from graphsage_src.my_models import *

from ZHMolGraph.ZHMolGraph_model import *


class ZHMolGraph():

    # Class Initialisation
    def __init__(self,

                 interactions_location=None,
                 interactions=None,
                 interaction_y_name='Y',

                 absolute_negatives_location=None,
                 absolute_negatives=None,

                 rnas_location=None,
                 rnas_dataframe=None,
                 rna_seq_name=None,
                 rna_embedding_name="normalized_embeddings",

                 proteins_location=None,
                 proteins_dataframe=None,
                 protein_seq_name=None,
                 protein_embedding_name="normalized_embeddings",



                 protvec_location=None,
                 protvec_model=None,

                 rna2vec_location=None,
                 rna2vec_model=None,

                 nodes_test=[],
                 nodes_validation=[],

                 edges_test=[],
                 edges_validation=[],

                 model_out_dir=None,

                 debug=False):

        # Set Variables
        self.interactions_location = interactions_location
        self.interactions = interactions
        self.interaction_y_name = interaction_y_name

        self.absolute_negatives_location = absolute_negatives_location
        self.absolute_negatives = absolute_negatives

        self.rnas_location = rnas_location
        self.rnas_dataframe = rnas_dataframe
        self.rna_seq_name = rna_seq_name
        self.rna_embedding_name = rna_embedding_name

        self.proteins_location = proteins_location
        self.proteins_dataframe = proteins_dataframe
        self.protein_seq_name = protein_seq_name
        self.protein_embedding_name = protein_embedding_name




        self.protvec_location = protvec_location
        self.protvec_model = protvec_model

        self.rna2vec_location = rna2vec_location
        self.rna2vec_model = rna2vec_model

        self.nodes_test = nodes_test
        self.nodes_validation = nodes_validation
        self.edges_test = edges_test
        self.edges_validation = edges_validation

        self.model_out_dir = model_out_dir

        self.debug = debug

        # Read In rnas
        if type(self.rnas_dataframe) == type(None):
            self.rnas_dataframe = self.read_input_files(self.rnas_location)

        # Read In Targets
        if type(self.proteins_dataframe) == type(None):
            self.proteins_dataframe = self.read_input_files(self.proteins_location)

        # Create rna Target Lists
        self.rna_list = list(self.rnas_dataframe[self.rna_seq_name])

        self.protein_list = list(self.proteins_dataframe[self.protein_seq_name])

        # Read In Interactions File
        if type(self.interactions) == type(None):
            self.interactions = self.read_input_files(self.interactions_location)

        # Read In Absolute Negatives File
        if type(self.absolute_negatives) == type(None):
            if type(self.absolute_negatives_location) != type(None):
                self.absolute_negatives = self.read_input_files(self.absolute_negatives_location)

        # Column Name Assertions
        assert self.rna_seq_name in self.interactions.columns, "Please ensure columns with seq Keys have the same name across all dataframes"
        assert self.rna_seq_name in self.rnas_dataframe.columns, "Please ensure columns with seq Keys have the same name across all dataframes"

        if self.nodes_test != []:
            assert self.rna_seq_name in self.nodes_test[
                0].columns, "Please ensure columns with seq Keys have the same name across all dataframes"
            assert self.rna_seq_name in self.nodes_validation[
                0].columns, "Please ensure columns with seq Keys have the same name across all dataframes"
            assert self.rna_seq_name in self.edges_test[
                0].columns, "Please ensure columns with seq Keys have the same name across all dataframes"
            assert self.rna_seq_name in self.edges_validation[
                0].columns, "Please ensure columns with seq Keys have the same name across all dataframes"

        assert self.protein_seq_name in self.interactions.columns, "Please ensure columns with Amino Acid Sequences have the same name across all dataframes"
        assert self.protein_seq_name in self.proteins_dataframe.columns, "Please ensure columns with Amino Acid Sequences have the same name across all dataframes"

        if self.nodes_test != []:
            assert self.protein_seq_name in self.nodes_test[
                0].columns, "Please ensure columns with Amino Acid Sequences have the same name across all dataframes"
            assert self.protein_seq_name in self.nodes_validation[
                0].columns, "Please ensure columns with Amino Acid Sequences have the same name across all dataframes"
            assert self.protein_seq_name in self.edges_test[
                0].columns, "Please ensure columns with Amino Acid Sequences have the same name across all dataframes"
            assert self.protein_seq_name in self.edges_validation[
                0].columns, "Please ensure columns withAmino Acid Sequences have the same name across all dataframes"



    ###################################################
    ############    General Functions      ############
    ###################################################

    def read_input_files(self, input_location):

        '''
        Reads in files into a dataframe given a file location. Currently works with CSV and Pickle files.

        Inputs :
            input_location : String - Location of file to read in - accepts only CSV and Pickle files
        Outputs :
            Pandas DatraFrame

        '''

        assert type(input_location) == type(""), 'Location should be of type str'

        if input_location.split('.')[-1] == 'pkl':
            with open(input_location, 'rb') as file:
                return pkl.load(file)

        elif input_location.split('.')[-1] == 'csv':
            return pd.read_csv(input_location)

        else:
            raise TypeError("Unknown input file type, only pkl and csv are supported")


    def dataframe_to_embed_array(self, interactions_df, rna_list, protein_list, rna_embed_len, normalized_rna_embeddings = None, normalized_protein_embeddings = None, include_true_label = True):

        '''
            Creates numpy arrays that can be fed into the model from interaction dataframes.

            Inputs :
                interactions_df : Pandas DataFrame - Pandas dataframe containing interactions
                rna_list : List - List of rna seq Keys
                protein_list : List - List of protein AA Sequences
                rna_embed_len : Integer - Length of rna embedding vector

            Outputs :
                X_0 : Numpy Array - Array with protein vectors
                X_1 : Numpy Array - Array with rna vectors
                Y :  Numpy Array - Array with true labels
        '''

        X_0_list = []
        X_1_list = []

        if type(normalized_protein_embeddings) == type(None):
            normalized_protein_embeddings = self.normalized_protein_embeddings

        if type(normalized_rna_embeddings) == type(None):
            normalized_rna_embeddings = self.normalized_rna_embeddings

        skipped_rnas = 0

        # Iterate over all rows in dataframe
        for idx, row in interactions_df.iterrows():

            # Get RNA and AA Sequence
            rna = row[self.rna_seq_name]
            protein = row[self.protein_seq_name]

            # Get rna index for this rna in rna_list
            try:
                rna_index = rna_list.index(rna)
            except:
                rna_index = -1

            # Get protein index for this protein in protein_list
            protein_index = protein_list.index(protein)

            # Index into protein embedding array and add to X_0
            X_0_list.append(normalized_protein_embeddings[protein_index])

            # If rna index not found, add random vector to X_1
            if rna_index == -1:
                X_1_list.append(np.random.randn(rna_embed_len,))
                skipped_rnas = skipped_rnas + 1
            else:
                # Index into rna embedding array and add to X_1
                try:
                    X_1_list.append(normalized_rna_embeddings[rna_index])
                # If rna index not found, add random vector to X_1
                except:
                    X_1_list.append(np.random.randn(rna_embed_len,))
                    skipped_rnas = skipped_rnas + 1

        # Convert lists to arrays
        X_0 = np.array(X_0_list)
        X_1 = np.array(X_1_list)

        if self.debug:
            print ("Number of rnas skipped : ", skipped_rnas)

        if include_true_label:
            Y = np.array(list(interactions_df['Y']))
            return X_0, X_1, Y
        else:
            return X_0, X_1



    ###################################################
    ############ Test/Validation Functions ############n
    ###################################################
    def get_benchmark_validation_ZHMolGraph_results(self, rna_embedding_length=640, protein_embedding_length=1024, dataset=None, 
        embedding_type=None, graphsage_embedding=1, result_file="ZHMolGraph_Line.csv"):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_index = torch.cuda.current_device()
        # print(device_index)
        # Iterate over 5 folds
        # Create Lists To Hold Information
        test_auc_ue = []
        test_aup_ue = []
        test_accuracy_ue = []
        test_precision_ue = []
        test_recall_ue = []
        test_mcc_ue = []
        test_spe_ue = []

        for run_number in tqdm(range(len(self.train_sets))):
            print(20*"——")
            print(f"Run_{run_number}")
            print(20*"——")


            graphsage_model_path = os.path.join(self.model_out_dir,
                                                'Run_' + str(run_number), "graphSage.pth")

            self.get_test_graphsage_embeddings(self.train_sets[run_number], self.test_sets[run_number], self.rnas,
                                          self.proteins, dataset, rna_embedding_length, protein_embedding_length,
                                          embedding_type=embedding_type, graphsage_embedding=graphsage_embedding, graphsage_model_path=graphsage_model_path)

            # self.get_graphsage_embeddings(self.train_sets[run_number], self.test_sets[run_number], self.rnas,
            #                               self.proteins, dataset, rna_embedding_length, protein_embedding_length,
            #                               embedding_type=embedding_type, graphsage_embedding=graphsage_embedding)

            self.normalized_protein_embeddings = self.graphsage_proteins_embeddings
            self.normalized_rna_embeddings = self.graphsage_rnas_embeddings


            self.rna_embed_len = self.normalized_rna_embeddings[0].shape[0]
            self.protein_embed_len = self.normalized_protein_embeddings[0].shape[0]





            # Reinitialise Model At Each Run
            # best_model = VecNet(target_embed_len=protein_embedding_length, rna_embed_len=rna_embedding_length)
            # print(f"best_model")
            # print(best_model)
            # print(best_model.state_dict())


            # 从检查点中加载模型状态字典

            best_model_path = os.path.join(self.model_out_dir, 'Run_' + str(run_number), f"VecNN_5_fold_Benchmark_Dataset_{dataset}.pth")

            best_model = torch.load(best_model_path)
            # if torch.cuda.device_count() > 1:
            #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            #     best_model = torch.nn.DataParallel(best_model)
            # print(f"loaded best_model")
            # print(best_model)
            # print(best_model.state_dict())
            best_model.to(device)
            with torch.no_grad():

                # Create Validation DataFrames For Each Run

                X_0_test_ue, X_1_test_ue, Y_test_actual_ue = self.dataframe_to_embed_array(
                    interactions_df=self.test_sets[run_number],
                    rna_list=self.rna_list,
                    protein_list=self.protein_list,
                    rna_embed_len=self.rna_embed_len)
                # print(X_0_test_ue)
                # print(X_0_test_ue.shape)
                # print(X_1_test_ue)


                # 计算分割点
                split_index = int(len(self.train_sets[run_number]) * 0.9)
                train_set = self.train_sets[run_number][:split_index]
                val_set = self.train_sets[run_number][split_index:]
                # For Each Epoch
                X_0_val_ue, X_1_val_ue, Y_val_actual_ue = self.dataframe_to_embed_array(
                    interactions_df=val_set,
                    rna_list=self.rna_list,
                    protein_list=self.protein_list,
                    rna_embed_len=self.rna_embed_len)
                # loss_last = float('inf')
                # loss = loss_last
                best_model.eval()  # Set model to evaluation mode
                # with torch.no_grad():
                    #Validation predictions

                Y_val_predictions_ue = best_model(torch.tensor(X_0_val_ue, dtype=torch.float32).to(device),
                                                  torch.tensor(X_1_val_ue, dtype=torch.float32).to(
                                                      device)).detach().cpu().numpy()

                # print("Y_val_predictions_ue")
                # print(Y_val_predictions_ue)
                # print("Y_val_actual_ue")
                # print(Y_val_actual_ue)


                # 根据验证集上的表现来选择概率的cutoff
                # 定义一组候选阈值
                thresholds = np.linspace(0, 1, 100)

                best_mcc = -1
                best_mcc_threshold = 0

                best_f1score = -1
                best_f1score_threshold = 0

                # 寻找最佳阈值
                for threshold in thresholds:
                    Y_val_predicted_labels = np.array(Y_val_predictions_ue > threshold, dtype=int)
                    mcc = matthews_corrcoef(Y_val_actual_ue, Y_val_predicted_labels)
                    # print(f"mcc: {mcc}, threshold: {threshold}")
                    # 将真实标签和预测标签转换为一维数组
                    Y_val_predicted_labels = np.ravel(Y_val_predicted_labels)
                    f1score = f1_score(Y_val_actual_ue, Y_val_predicted_labels)
                    # print(Y_val_predicted_labels)
                    # print(Y_val_actual_ue)
                    if mcc > best_mcc:
                        best_mcc = mcc
                        best_mcc_threshold = threshold
                    if f1score > best_f1score:
                        best_f1score = f1score
                        best_f1score_threshold = threshold
                # print(f"Best MCC: {best_mcc}")
                # print(f"Best MCC Threshold: {best_mcc_threshold}")
                #
                # print(f"Best f1score: {best_f1score}")
                # print(f"Best f1score Threshold: {best_f1score_threshold}")

                cutoff = best_mcc_threshold

                # Testing predictions
                Y_test_predictions_ue = best_model(torch.tensor(X_0_test_ue, dtype=torch.float32).to(device),
                                                   torch.tensor(X_1_test_ue, dtype=torch.float32).to(device)).detach().cpu().numpy()

                # print(X_0_test_ue)
                # print(X_1_test_ue)
                # print("Y_test_predictions_ue")
                # print(Y_test_predictions_ue)

                curr_test_auc = roc_auc_score(Y_test_actual_ue, Y_test_predictions_ue)
                curr_test_aup = average_precision_score(Y_test_actual_ue, Y_test_predictions_ue)
                # 将预测概率转化为二分类预测（例如，大于0.5的为正类，小于等于0.5的为负类）

                Y_test_predictions_ue = np.array(Y_test_predictions_ue)
                Y_test_actual_ue = np.array(Y_test_actual_ue)



                Y_test_predictions_labels = np.array(Y_test_predictions_ue > cutoff, dtype=int)
                # print(Y_test_predictions_ue)
                # print(Y_test_actual_ue)
                # Y_test_predictions_labels = np.array(Y_test_predictions_ue > np.median(Y_test_predictions_ue),
                #                                     dtype=int)
                #print(Y_test_predictions_labels)
                # 计算accuracy值
                curr_test_accuracy = accuracy_score(Y_test_actual_ue, Y_test_predictions_labels)
                # 计算precision值
                curr_test_precision = precision_score(Y_test_actual_ue, Y_test_predictions_labels)
                # 计算recall值
                curr_test_recall = recall_score(Y_test_actual_ue, Y_test_predictions_labels)
                # 计算MCC值
                curr_test_mcc = matthews_corrcoef(Y_test_actual_ue, Y_test_predictions_labels)
                # 计算混淆矩阵
                cm = confusion_matrix(Y_test_actual_ue, Y_test_predictions_labels)

                # 提取混淆矩阵的值
                tn, fp, fn, tp = cm.ravel()

                # 计算 Specificity
                curr_test_spe = tn / (tn + fp)

                test_aup_ue.append(curr_test_aup)
                test_auc_ue.append(curr_test_auc)
                test_accuracy_ue.append(curr_test_accuracy)
                test_precision_ue.append(curr_test_precision)
                test_recall_ue.append(curr_test_recall)
                test_mcc_ue.append(curr_test_mcc)
                test_spe_ue.append(curr_test_spe)

                output_string = ""
                # Print Stuff
                # output_string = output_string + "AUROC : " + str(
                #     np.round(test_auc_ue[-1], 3)) + "\n"
                # output_string = output_string + "AUPRC : " + str(
                #     np.round(test_aup_ue[-1], 3)) + "\n"
                output_string = output_string + "accuracy : " + str(np.round(curr_test_accuracy, 3)) + "\n"
                output_string = output_string + "sensitivity : " + str(np.round(curr_test_recall, 3)) + "\n"
                output_string = output_string + "specificity : " + str(np.round(curr_test_spe, 3)) + "\n"
                output_string = output_string + "precision : " + str(np.round(curr_test_precision, 3)) + "\n"
                output_string = output_string + "mcc : " + str(np.round(curr_test_mcc, 3)) + "\n"
                output_string = output_string + "tn : " + str(tn) + "\n"
                output_string = output_string + "fp : " + str(fp) + "\n"
                output_string = output_string + "fn : " + str(fn) + "\n"
                output_string = output_string + "tp : " + str(tp) + "\n"

                print(25 * "——")
                print(f"Performance of Run_{run_number}")
                print(25 * "——")

                # print(output_string)
                output_dir = f'result/{dataset}'
                os.makedirs(output_dir, exist_ok=True)

                with open(f"{output_dir}/Run_{run_number}", "a") as file:
                    output_result_text = ""
                    output_result_text = output_result_text + "accuracy : " + "{:.3f}".format(curr_test_accuracy) + ", "
                    output_result_text = output_result_text + "sensitivity : " + "{:.3f}".format(curr_test_recall)+ ", "
                    output_result_text = output_result_text + "specificity : " + "{:.3f}".format(curr_test_spe) + ", "
                    output_result_text = output_result_text + "precision : " +  "{:.3f}".format(curr_test_precision) + ", "
                    output_result_text= output_result_text + "mcc : " + "{:.3f}".format(curr_test_mcc) + "\n"

                    file.write(output_result_text)

        print(25 * "——")
        print(f"Validation Performance of Dataset {dataset}: ")
        print(25 * "——")

        # print("Best Model Suffix : ", self.model_name_index[model_name][best_model])
        # print("AUROC : ", np.round(np.mean(test_auc_ue), 3), "+/-", np.std(test_auc_ue))
        # print("AUPRC : ", np.round(np.mean(test_aup_ue), 3), "+/-", np.std(test_aup_ue))
        print("Accuracy : ", np.round(np.mean(test_accuracy_ue), 3), "+/-", np.std(test_accuracy_ue))
        print("Sensitivity : ", np.round(np.mean(test_recall_ue), 3), "+/-", np.std(test_recall_ue))
        print("Specificity : ", np.round(np.mean(test_spe_ue), 3), "+/-", np.std(test_spe_ue))
        print("Precision : ", np.round(np.mean(test_precision_ue), 3), "+/-", np.std(test_precision_ue))
        print("MCC : ", np.round(np.mean(test_mcc_ue), 3), "+/-", np.std(test_mcc_ue))

        test_accuracy_ue_df = pd.DataFrame({'Acc': np.round(test_accuracy_ue, 3)})
        test_recall_ue_df = pd.DataFrame({'Sen': np.round(test_recall_ue, 3)})
        test_spe_ue_df = pd.DataFrame({'Spe': np.round(test_spe_ue, 3)})
        test_precision_ue_df = pd.DataFrame({'Pre': np.round(test_precision_ue, 3)})
        test_mcc_ue_df = pd.DataFrame({'MCC': np.round(test_mcc_ue, 3)})
        performance_df = pd.concat([test_accuracy_ue_df, test_recall_ue_df, test_spe_ue_df, test_precision_ue_df, test_mcc_ue_df], axis=1)
        # print(performance_df)
        # print("写入result文件夹")
        # 获取当前工作目录
        current_directory = os.getcwd()
        performance_df_file = os.path.join(current_directory, "Result", f"{dataset}_" + result_file)

        performance_df.to_csv(performance_df_file, index=False, sep=',')  # 使用相对路径


    #### 得到未见测试集的测试结果 ####

    def get_TheNovel_test_results(self, graphsage_path=None, test_dataframe=None, model_dataset=None,
                                  unseen_dataset=None, rna_vector_length=None, protein_vector_length=None, rnas=None, proteins=None,
                                            result_path=None, embedding_type=None):



        if model_dataset == None:
            sys.exit("请输入正确的数据集！")

        if type(test_dataframe) == type(None):
            sys.exit("请输入测试集！")
        self.averaged_results = {}
        plot_div_counter = 0


        test_auc_ue = []
        test_aup_ue = []
        test_accuracy_ue = []
        test_precision_ue = []
        test_recall_ue = []
        test_mcc_ue = []
        test_spe_ue = []





        for run in range(5):

            # print(20*"——")
            # print(f"Run_{run}")
            # print(20*"——")
            ### 获取graphsage的嵌入
            graphsage_model_path = graphsage_path+'/Run_' + str(run) + '/graphSage.pth'
            # print(os.getcwd())
            self.get_unseen_graphsage_embeddings(model_path=graphsage_model_path, model_dataset=model_dataset,
                                                          unseen_dataset=unseen_dataset, embedding_type=embedding_type,
                                                 rna_vector_length=rna_vector_length, protein_vector_length=protein_vector_length,
                                                 rnas=rnas, proteins=proteins)
            self.normalized_protein_embeddings = self.graphsage_proteins_embeddings
            self.normalized_rna_embeddings = self.graphsage_rnas_embeddings

            # print(f'self.protein_list: {self.protein_list}')
            X_0_test, X_1_test, Y_test_actual = self.dataframe_to_embed_array(
                interactions_df=test_dataframe,
                rna_list=self.rna_list,
                protein_list=self.protein_list,
                rna_embed_len=rna_vector_length+100)
            # print(X_0_test)
            # print(X_0_test.shape)
            # print(X_1_test)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_index = torch.cuda.current_device()
            #print(device_index)


            best_model_path = os.path.join(self.model_out_dir, 'Run_' + str(run), f"VecNN_5_fold_Benchmark_Dataset_{model_dataset}.pth")
            # 加载模型
            # best_vecnet_model = VecNet(target_embed_len=protein_vector_length, rna_embed_len=rna_vector_length)


            # 从检查点中加载模型状态字典
            best_vecnet_model = torch.load(best_model_path)
            # if torch.cuda.device_count() > 1:
            #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            #     best_vecnet_model = torch.nn.DataParallel(best_vecnet_model)
            best_vecnet_model.to(device)

            # 确保模型处于评估模式（不进行梯度计算）
            best_vecnet_model.eval()
            # print(run)
            # print(best_vecnet_model)
            Y_test_predictions = best_vecnet_model(torch.tensor(X_0_test, dtype=torch.float32).to(device),
                                          torch.tensor(X_1_test, dtype=torch.float32).to(
                                              device)).detach().cpu().numpy()


            # print(Y_test_predictions)
            # print('真实标签')
            # print(Y_test_actual)

            # 使用reshape将其转置为列向量
            Y_test_actual_T = np.reshape(Y_test_actual, (len(Y_test_actual), 1))
            # Y_test_actual_T = Y_test_actual_T.T

            # print(Y_test_actual_T)

            # 使用concatenate函数合并列
            Y_test_predictions_actual = np.concatenate((Y_test_predictions, Y_test_actual_T), axis=1)
            # print(Y_test_predictions_actual)

            # 输出地址
            Y_test_predictions_actual_path = 'Result/UnseenNodeTopPrediction/' + model_dataset + '_' + embedding_type  + '/'
            os.makedirs(Y_test_predictions_actual_path, exist_ok=True)
            Y_test_predictions_actual_path_file = Y_test_predictions_actual_path + 'Run' + str(run) + '.csv'
            np.savetxt(Y_test_predictions_actual_path_file, Y_test_predictions_actual, delimiter=',')




            # cutoff = 0.09
            # Y_test_predictions_labels = np.array(Y_test_predictions > cutoff, dtype=int)
            # print(Y_test_predictions_labels)
            # 计算混淆矩阵
            # cm = confusion_matrix(Y_test_actual, Y_test_predictions_labels)

            # 提取混淆矩阵的值
            # tn, fp, fn, tp = cm.ravel()
            # print(f"tp:{tp}")
            # print(f"fn:{fp}")
            # print(f"tp:{tn}")
            # print(f"fn:{fn}")
            # 计算accuracy值
            curr_test_auc = roc_auc_score(Y_test_actual, Y_test_predictions)
            curr_test_aup = average_precision_score(Y_test_actual, Y_test_predictions)
            # curr_test_accuracy = accuracy_score(Y_test_actual, Y_test_predictions_labels)
            # 计算precision值
            # curr_test_precision = precision_score(Y_test_actual, Y_test_predictions_labels)
            # 计算recall值
            # curr_test_recall = recall_score(Y_test_actual, Y_test_predictions_labels)
            # 计算MCC值
            # curr_test_mcc = matthews_corrcoef(Y_test_actual, Y_test_predictions_labels)

            # 计算 Specificity
            # curr_test_spe = tn / (tn + fp)

            output_string = ""
            # Print Stuff
            output_string = output_string + "AUROC : " + str(
                np.round(curr_test_auc, 3)) + "\n"
            output_string = output_string + "AUPRC : " + str(
                np.round(curr_test_aup, 3)) + "\n"
            # output_string = output_string + "accuracy : " + str(np.round(curr_test_accuracy, 3)) + "\n"
            # output_string = output_string + "Sensitivity : " + str(np.round(curr_test_recall, 3)) + "\n"
            # output_string = output_string + "specificity : " + str(np.round(curr_test_spe, 3)) + "\n"
            # output_string = output_string + "precision : " + str(np.round(curr_test_precision, 3)) + "\n"
            # output_string = output_string + "mcc : " + str(np.round(curr_test_mcc, 3)) + "\n"
            # output_string = output_string + "tn : " + str(tn) + "\n"
            # output_string = output_string + "fp : " + str(fp) + "\n"
            # output_string = output_string + "fn : " + str(fn) + "\n"
            # output_string = output_string + "tp : " + str(tp) + "\n"
            test_aup_ue.append(curr_test_aup)
            test_auc_ue.append(curr_test_auc)
            # test_accuracy_ue.append(curr_test_accuracy)
            # test_precision_ue.append(curr_test_precision)
            # test_recall_ue.append(curr_test_recall)
            # test_mcc_ue.append(curr_test_mcc)
            # test_spe_ue.append(curr_test_spe)

            result_auc_aup_dict = {'AUROC': curr_test_auc, 'AUPRC': curr_test_aup}
            result_auc_aup = pd.DataFrame(result_auc_aup_dict, index=[run])


            result_auc_aup_path = result_path + str(run) + '.csv'

            result_auc_aup.to_csv(result_auc_aup_path, sep=',')

            # print(25 * "——")
            # print(f"Performance of Run_{run}")
            # print(25 * "——")
            # print(output_string)


        test_output_string = ""
        # Print Stuff
        test_output_string = test_output_string + "AUROC : " + str(
            np.round(np.mean(test_auc_ue), 3)) + "\n"
        test_output_string = test_output_string + "AUPRC : " + str(
            np.round(np.mean(test_aup_ue), 3)) + "\n"
        # test_output_string = test_output_string + "accuracy : " + str(np.round(np.mean(test_accuracy_ue), 3)) + "\n"
        # test_output_string = test_output_string + "sensitivity : " + str(np.round(np.mean(test_recall_ue), 3)) + "\n"
        # test_output_string = test_output_string + "specificity : " + str(np.round(np.mean(test_spe_ue), 3)) + "\n"
        # test_output_string = test_output_string + "precision : " + str(np.round(np.mean(test_precision_ue), 3)) + "\n"
        # test_output_string = test_output_string + "mcc : " + str(np.round(np.mean(test_mcc_ue), 3)) + "\n"
        # print(25 * "——")
        # print(f"Validation Performance of Dataset TheNovel: ")
        # print(25 * "——")
        # print(test_output_string)



    def get_test_graphsage_embeddings(self, train_set, test_set, rnas, proteins, dataset, rnas_length, proteins_length, embedding_type, graphsage_embedding=1, graphsage_model_path=None):
        '''
            Reads in graphsage embeddings for all test RNA and proteins

            Inputs :
            train_set: The training set of training interactions list [RNA_aa_code, target_aa_code, Y]
            test_set: The testing set of testing interactions list [RNA_aa_code, target_aa_code, Y]
            rnas: The corresponding sequence of rna and the normalized_embedding from RNA-FM [RNA_aa_code, normalized_embeddings]
            proteins: The corresponding sequence of protein and the normalized_embedding from ProtTrans [target_aa_code, normalized_embeddings]

            Outputs :
            self.graphsage_rna_embeddings: The embeddings of rnas derived from graphsage neural network model [RNA_aa_code, graphsage_embeddings]
            self.graphsage_protein_embeddings: The embeddings of proteins derived from graphsage neural network model [target_aa_code, graphsage_embeddings]

        '''
        ### 生成用于graphsage训练的四个文件, 包含所有相互作用：RPI2241_total_interactions_seq_list.txt；包含所有节点的特征：RPI2241_feats.txt；
        # graphsage训练集中的所有相互作用：RPI2241_graphsage_train_interactions.txt; graphsage测试集中的所有相互作用：RPI2241_graphsage_test_interactions.txt
        train_interaction_df = pd.concat([train_set['RNA_aa_code'], train_set['target_aa_code']], axis=1)  # 训练集中所有相互作用
        test_interaction_df = pd.concat([test_set['RNA_aa_code'], test_set['target_aa_code']], axis=1)  # 测试集中所有相互作用

        if not os.path.exists(dataset + '_' + embedding_type + '_graphsage_dataset/'):
            # 如果不存在，创建文件夹
            os.makedirs(dataset + '_' + embedding_type + '_graphsage_dataset/')

        # 生成所有的相互作用：RPI2241_total_interactions_seq_list.txt
        total_interaction_df = pd.concat([train_interaction_df, test_interaction_df], axis=0)  # 所有的相互作用
        total_interaction_df.to_csv(
            dataset + '_' + embedding_type + '_graphsage_dataset/' + dataset + '_total_interactions_seq_list.txt',
            sep='\t', index=False, header=False)

        # 生成包含所有节点特征的文件

        # 将节点的embedding变成np矩阵
        rna_embeddings = rnas['normalized_embeddings']
        rna_array = np.zeros((len(rnas['normalized_embeddings']), rnas_length))

        protein_embeddings = proteins['normalized_embeddings']
        protein_array = np.zeros((len(proteins['normalized_embeddings']), proteins_length))

        # 使用 for 循环逐行赋值
        for i in range(len(rnas['normalized_embeddings'])):
            rna_array[i, :] = rna_embeddings.iloc[i]

        # 使用 for 循环逐行赋值
        for i in range(len(proteins['normalized_embeddings'])):
            protein_array[i, :] = protein_embeddings.iloc[i]
        # print(rna_array)
        # print(rna_array.shape)

        if graphsage_embedding == 1:
        #     # # 把proteins的特征映射到640维
        #     # # 创建一个 PCA 模型，指定目标维度为 k
        #     # k = 640
        #     # pca = PCA(n_components=k)
        #     #
        #     # # 对数据进行拟合和转换
        #     # compressed_protein_array = pca.fit_transform(protein_array)
        #     # # print(compressed_protein_array)
        #
        #     ### 制作 RNA 的特征文件
        #     # 创建一个空的 RNA DataFrame
        #     if rnas_length < proteins_length:
        #         columns = ['nodes'] + [f'fea_{i}' for i in range(1, proteins_length + 1)]
        #     elif rnas_length >= proteins_length:
        #         columns = ['nodes'] + [f'fea_{i}' for i in range(1, rnas_length + 1)]
        #     RNA_feat = pd.DataFrame(columns=columns)
        #
        #     # 打印结果
        #     print(RNA_feat)
        #
        #     for i in range(rnas.shape[0]):
        #         RNA_data = []
        #         rna_seq = rnas['RNA_aa_code'][i]
        #         RNA_data.append(rna_seq)
        #
        #         rna_embedding = rna_array[i, :]
        #         # print(rna_embedding)
        #         for i in range(len(rna_embedding)):
        #             RNA_data.append(rna_embedding[i])
        #
        #         if rnas_length < proteins_length:
        #             RNA_data = RNA_data + [0] * max(0, proteins_length + 1 - len(RNA_data))
        #
        #         # print(RNA_data)
        #         RNA_feat.loc[len(RNA_feat)] = RNA_data
        #     # print(RNA_feat)
        #     print("load RNA_feat")
        #     ### 制作 protein 的特征文件
        #     # 创建一个空的 protein DataFrame
        #     if rnas_length < proteins_length:
        #         columns = ['nodes'] + [f'fea_{i}' for i in range(1, proteins_length + 1)]
        #     elif rnas_length >= proteins_length:
        #         columns = ['nodes'] + [f'fea_{i}' for i in range(1, rnas_length + 1)]
        #     protein_feat = pd.DataFrame(columns=columns)
        #
        #     # 打印结果
        #     # print(protein_feat)
        #     for i in range(proteins.shape[0]):
        #         protein_data = []
        #         protein_seq = proteins['target_aa_code'][i]
        #         protein_data.append(protein_seq)
        #
        #         # protein_embedding = compressed_protein_array[i, :]
        #         protein_embedding = protein_array[i, :]
        #
        #         for i in range(len(protein_embedding)):
        #             protein_data.append(protein_embedding[i])
        #
        #         if rnas_length > proteins_length:
        #             protein_data = protein_data + [0] * max(0, rnas_length + 1 - len(protein_data))
        #
        #         protein_feat.loc[len(protein_feat)] = protein_data
        #     # print(protein_feat)
        #     print("load protein_feat")
        #     total_feat = pd.concat([RNA_feat, protein_feat], axis=0)
        #     total_feat = total_feat.reset_index(drop=True)
        #     print(total_feat.shape)
        #     # print(total_feat)
        #     print("load total_feat")
        #     ### 写入到文件
        #
        #     output_feats_file = dataset + '_' + embedding_type + '_graphsage_dataset/' + dataset + '_feats.txt'
        #     total_feat.to_csv(output_feats_file, sep='\t', index=False, header=False)

            # 获取训练集里的所有正样本
            positive_train_interaction_df = train_set[train_set['Y'] == 1]

            # 生成graphsage训练集中的所有相互作用：RPI2241_graphsage_train_interactions.txt;
            graphsage_train_interaction_df = pd.concat(
                [positive_train_interaction_df['RNA_aa_code'], positive_train_interaction_df['target_aa_code']], axis=1)
            graphsage_train_interaction_df.to_csv(
                dataset + '_' + embedding_type + '_graphsage_dataset/' + dataset + '_graphsage_train_interactions.txt',
                sep='\t', index=False, header=False)
            # print(graphsage_train_interaction_df)

            # 获取训练集里的所有负样本
            negative_train_interaction_df = train_set[train_set['Y'] == 0]
            negative_train_interaction_df = pd.concat(
                [negative_train_interaction_df['RNA_aa_code'], negative_train_interaction_df['target_aa_code']], axis=1)

            # 生成graphsage测试集中的所有相互作用：RPI2241_graphsage_test_interactions.txt
            graphsage_test_interaction_df = pd.concat([test_interaction_df, negative_train_interaction_df], axis=1)
            # print(graphsage_test_interaction_df)
            graphsage_test_interaction_df.to_csv(
                dataset + '_' + embedding_type + '_graphsage_dataset/' + dataset + '_graphsage_test_interactions.txt',
                sep='\t', index=False, header=False)

            # print(train_interaction_df)
            # print(test_interaction_df)
            # print(total_interaction_df)
            # print(positive_train_interaction_df)

            ### 训练graphsage模型
            dataset = dataset
            self.graphsage_embeddings = self.run_graphsage_model(dataSet=dataset, seed=64, cuda=True,
                                                                      config='./graphsage_src/experiments.conf',
                                                                      embedding_type=embedding_type, graphsage_model_path=graphsage_model_path)
            graphsage_embeddings = self.graphsage_embeddings
            # print(f'graphsage_embeddings.shape{graphsage_embeddings.shape}')
            graphsage_rnas_embeddings = graphsage_embeddings[:len(rnas)]
            # print(f'graphsage_rnas_embeddings.shape{graphsage_rnas_embeddings.shape}')
            # print(f'rnas的长度：{len(rnas)}')
            graphsage_proteins_embeddings = graphsage_embeddings[len(rnas):len(proteins) + len(rnas)]
            # self.graphsage_rnas_embeddings = graphsage_rnas_embeddings
            # self.graphsage_proteins_embeddings = graphsage_proteins_embeddings
            # self.graphsage_rnas_embeddings = rna_array
            # self.graphsage_proteins_embeddings = protein_array
            # print(graphsage_rnas_embeddings.shape)
            # print(rna_array.shape)
            self.graphsage_rnas_embeddings = np.concatenate((graphsage_rnas_embeddings, rna_array), axis=1)
            self.graphsage_proteins_embeddings = np.concatenate((graphsage_proteins_embeddings, protein_array), axis=1)
        elif graphsage_embedding == 0:
            self.graphsage_rnas_embeddings = rna_array
            self.graphsage_proteins_embeddings = protein_array
        else:
            print("请输入正确的graphsage_embedding的参数!")

    def get_unseen_graphsage_embeddings(self, config='./graphsage_src/experiments.conf',
                                        model_path=None, model_dataset=None, unseen_dataset=None,
                                        rna_vector_length=None, protein_vector_length=None,
                                        rnas=None, proteins=None, embedding_type=None, 
                                        test_interactions=None):
        '''
        获取未见过节点的网络上的embeddings
        '''
        # print(os.getcwd())

        #cuda = True
        #device = torch.device("cuda" if cuda else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_index = torch.cuda.current_device()
        # print(device_index)
        # print('DEVICE:', device)

        ### 输入模型
        model_path = model_path
        graphsage = torch.load(model_path,weights_only=False)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        #     graphsage = torch.nn.DataParallel(graphsage)

        #graphsage = torch.load(model_path, weights_only=False, map_location="cpu")
        # print(graphsage)

        graphsage.to(device)

        ### 导入训练集

        node_map, feat_data = graphsage.dataCenter.load_dataSet(model_dataset, embedding_type)
        # print(f"node_map'\n'{type(node_map)}")
        #
        # print(f"feat_data'\n'{type(feat_data)}")
        #
        # print(f"feat_data'\n'{feat_data.shape}")

        output_feats_file = unseen_dataset + '_graphsage_dataset/' + unseen_dataset + '_feats' + '.txt'

        test_rnas = rnas
        test_proteins = proteins
        # 将节点的embedding变成np矩阵
        rna_embeddings = test_rnas['normalized_embeddings']
        rna_array = np.zeros((len(test_rnas['normalized_embeddings']), rna_vector_length))
        # print(rna_embeddings)
        protein_embeddings = test_proteins['normalized_embeddings']
        protein_array = np.zeros((len(test_proteins['normalized_embeddings']), protein_vector_length))
        # print(protein_embeddings)
        # 使用 for 循环逐行赋值
        for i in range(len(test_rnas['normalized_embeddings'])):
            # print(rna_embeddings.iloc[i].shape)
            rna_array[i, :] = rna_embeddings.iloc[i]

        # 使用 for 循环逐行赋值
        for i in range(len(test_proteins['normalized_embeddings'])):
            protein_array[i, :] = protein_embeddings.iloc[i]


        if not os.path.exists(output_feats_file):
            ### 制作新测试集特征所需要的文件
            unseen_dataset = unseen_dataset
            unseen_dataset_path = 'data/interactions/' + 'dataset_RPI_' + unseen_dataset + '_RP.csv'
            if os.path.exists(unseen_dataset_path):
                unseen_interactions = pd.read_csv(unseen_dataset_path, sep=',')
            else:
                unseen_interactions = test_interactions

            # print(unseen_interactions)
            total_interaction_df = pd.concat(
                [unseen_interactions['RNA_aa_code'], unseen_interactions['target_aa_code']], axis=1)
            # print(total_interaction_df)

            ### 导入测试集预训练的嵌入 ###
            # Read In rnas and proteins dataframes to pass to AIBind after changing column names
            # with open('data/sars-busters/Mol2Vec/NPI_' + unseen_dataset + '_rnafm_embed_normal.pkl', 'rb') as file:
            #     test_rnas = pkl.load(file)
            #
            # with open('data/sars-busters/Mol2Vec/NPI_' + unseen_dataset + '_proteinprottrans_embed_normal.pkl', 'rb') as file:
            #     test_proteins = pkl.load(file)
            #
            # print(test_rnas)
            # print(len(test_rnas))
            # print(test_proteins)
            # print(type(test_proteins))

            ### 生成用于graphsage训练的四个文件, 包含所有相互作用：RPI2241_total_interactions_seq_list.txt；包含所有节点的特征：RPI2241_feats.txt；
            # graphsage训练集中的所有相互作用：RPI2241_graphsage_train_interactions.txt; graphsage测试集中的所有相互作用：RPI2241_graphsage_test_interactions.txt

            if not os.path.exists(unseen_dataset + '_graphsage_dataset/'):
                # 如果不存在，创建文件夹
                os.makedirs(unseen_dataset + '_graphsage_dataset/')


            total_interaction_df.to_csv(unseen_dataset + '_graphsage_dataset/' + unseen_dataset + '_total_interactions_seq_list' + '.txt',
                                        sep='\t', index=False, header=False)

            # columns = ['nodes'] + [f'fea_{i}' for i in range(1, 1024 + 1)]
            # RNA_feat = pd.DataFrame(columns=columns)
            rnas_length = rna_vector_length
            proteins_length =protein_vector_length
            if rnas_length < proteins_length:
                columns = ['nodes'] + [f'fea_{i}' for i in range(1, proteins_length + 1)]
            elif rnas_length >= proteins_length:
                columns = ['nodes'] + [f'fea_{i}' for i in range(1, rnas_length + 1)]
            RNA_feat = pd.DataFrame(columns=columns)

            # 打印结果
            # print(RNA_feat)

            for i in range(test_rnas.shape[0]):
                RNA_data = []
                rna_seq = test_rnas['RNA_aa_code'][i]
                RNA_data.append(rna_seq)

                rna_embedding = rna_array[i, :]

                for i in range(len(rna_embedding)):
                    RNA_data.append(rna_embedding[i])

                if rnas_length < proteins_length:
                    RNA_data = RNA_data + [0] * max(0, proteins_length + 1 - len(RNA_data))
                # RNA_data = RNA_data + [0] * max(0, 1025 - len(RNA_data))
                # print(RNA_data)
                RNA_feat.loc[len(RNA_feat)] = RNA_data
            # print(RNA_feat)
            # print("load RNA_feat")
            ### 制作 protein 的特征文件
            # 创建一个空的 protein DataFrame
            # columns = ['nodes'] + [f'fea_{i}' for i in range(1, 1024 + 1)]
            if rnas_length < proteins_length:
                columns = ['nodes'] + [f'fea_{i}' for i in range(1, proteins_length + 1)]
            elif rnas_length >= proteins_length:
                columns = ['nodes'] + [f'fea_{i}' for i in range(1, rnas_length + 1)]
            protein_feat = pd.DataFrame(columns=columns)

            # 打印结果
            # print(protein_feat)
            for i in range(test_proteins.shape[0]):
                protein_data = []
                protein_seq = test_proteins['target_aa_code'][i]
                protein_data.append(protein_seq)

                # protein_embedding = compressed_protein_array[i, :]
                protein_embedding = protein_array[i, :]

                for i in range(len(protein_embedding)):
                    protein_data.append(protein_embedding[i])
                if rnas_length > proteins_length:
                    protein_data = protein_data + [0] * max(0, rnas_length + 1 - len(protein_data))

                protein_feat.loc[len(protein_feat)] = protein_data
            # print(protein_feat)
            # print("load protein_feat")
            total_feat = pd.concat([RNA_feat, protein_feat], axis=0)
            total_feat = total_feat.reset_index(drop=True)
            # print(total_feat.shape)
            # print(total_feat)

            # print("load total_feat")
            ### 写入到文件

            total_feat.to_csv(output_feats_file, sep='\t', index=False, header=False)

        feat_data_list = feat_data.tolist()
        # print('输出feat_data_list')
        # print(type(feat_data_list))




        config = pyhocon.ConfigFactory.parse_file(config)

        ds = model_dataset
        dataCenter = DataCenter(config)
        dataCenter.load_dataSet(ds, embedding_type=embedding_type) #运行dataCenter

        graphsage.dataCenter = dataCenter
        
        unseen_indexs, feat_data = graphsage.dataCenter.load_unseen_dataSet(unseen_dataset, graphsage.adj_lists,
                                                                            node_map, feat_data_list)

        # print('输出feat_data')
        # print(feat_data)
        # print(feat_data.shape)
        # print(type(feat_data))

        raw_features = torch.FloatTensor(feat_data).to(device)
        graphsage.raw_features = raw_features

        graphsage.to(device)
        graphsage.eval()
        embs = get_unseen_gnn_embeddings(graphsage, graphsage.dataCenter, unseen_dataset, graphsage.adj_lists)
        # print(f"嵌入的尺寸: {embs.shape}")
        # file_path = 'test_embeddings.csv'
        # with open(file_path, 'w', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file, delimiter='\t')
        #     for row in embs:
        #         csv_writer.writerow(row.tolist())

        graphsage_embeddings = embs.cpu().numpy()

        graphsage_rnas_embeddings = graphsage_embeddings[:len(test_rnas)]
        graphsage_proteins_embeddings = graphsage_embeddings[len(test_rnas):len(test_proteins) + len(test_rnas)]
        # self.graphsage_rnas_embeddings = graphsage_rnas_embeddings
        # self.graphsage_proteins_embeddings = graphsage_proteins_embeddings
        # self.graphsage_rnas_embeddings = rna_array
        # self.graphsage_proteins_embeddings = protein_array
        self.graphsage_rnas_embeddings = np.concatenate((graphsage_rnas_embeddings, rna_array), axis=1)
        self.graphsage_proteins_embeddings = np.concatenate((graphsage_proteins_embeddings, protein_array), axis=1)




    def run_graphsage_model(self, dataSet=None, seed=64, cuda=False, config='./graphsage_src/experiments.conf', embedding_type=None, graphsage_model_path=None):

        if dataSet==None:
            sys.exit("请输入正确数据集!")
        if torch.cuda.is_available():
            if not cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                device_id = torch.cuda.current_device()
                print('using device', device_id, torch.cuda.get_device_name(device_id))
            cuda = torch.cuda.is_available()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_index = torch.cuda.current_device()
        # print(device_index)
        # print('DEVICE:', device)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        config = pyhocon.ConfigFactory.parse_file(config)

        ds = dataSet
        dataCenter = DataCenter(config)
        dataCenter.load_dataSet(ds, embedding_type=embedding_type) #运行dataCenter
        features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)
        # print(f'feature in dataCenter: {features}')
        cuda = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('DEVICE:', device)

        ### 输入模型
        model_path = graphsage_model_path
        graphSage = torch.load(model_path)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        #     graphSage = torch.nn.DataParallel(graphSage)
        # print(graphSage)

        graphSage.to(device)
        graphSage.eval()


        # print("#############TEST###########")
        graphSage.eval()
        embs = get_gnn_embeddings(graphSage, dataCenter, ds, getattr(dataCenter, ds + '_adj_lists'))
        # print(f'Output embs: {embs}')
        # print(f'Output embs shape: {embs.shape}')
        self.graphsage_embeddings = embs.cpu().numpy()
        self.graphSage = graphSage
        return self.graphsage_embeddings

    def run_graphsage_experiment(self, dataSet=None, agg_func='MEAN', epochs=3, b_sz=20, seed=64, cuda=False,
                                 gcn=False, learn_method='unsup', unsup_loss='margin', max_vali_f1=0, name='debug',
                                 config='./graphsage_src/experiments.conf',embedding_type=None):

        if dataSet==None:
            sys.exit("请输入正确数据集!")
        if torch.cuda.is_available():
            if not cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                device_id = torch.cuda.current_device()
                print('using device', device_id, torch.cuda.get_device_name(device_id))
            cuda = torch.cuda.is_available()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_index = torch.cuda.current_device()
        # print(device_index)
        # print('DEVICE:', device)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        config = pyhocon.ConfigFactory.parse_file(config)

        ds = dataSet
        dataCenter = DataCenter(config)
        dataCenter.load_dataSet(ds, embedding_type=embedding_type) #运行dataCenter
        features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)
        print(f'feature in dataCenter: {features}')

        graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], dataCenter,
                              features, getattr(dataCenter, ds + '_adj_lists'), device, gcn=gcn,
                              agg_func=agg_func)
        graphSage.to(device)
        ### 直接输出，再保存graphsage的模型
        print(f"nn structure: {graphSage}")
        #print(graphSage.adj_lists)
        unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds + '_adj_lists'), getattr(dataCenter, ds + '_train'),
                                             device)

        if learn_method == 'sup':
            print('GraphSage with Supervised Learning')
        elif learn_method == 'plus_unsup':
            print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
        else:
            print('GraphSage with Net Unsupervised Learning')
        for epoch in range(epochs):
            print('----------------------EPOCH %d-----------------------' % epoch)
            graphSage = apply_model(dataCenter, ds, graphSage, unsupervised_loss, b_sz,
                                    unsup_loss, device, learn_method)

        print("#############TEST###########")
        graphSage.eval()
        embs = get_gnn_embeddings(graphSage, dataCenter, ds, getattr(dataCenter, ds + '_adj_lists'))
        self.graphsage_embeddings = embs.cpu().numpy()
        self.graphSage = graphSage
        return self.graphsage_embeddings

    #############################################################
    ###########          get embeddings              ############
    #############################################################
    # Get RNA Embeddings From RNAFM
    def get_rnafm_embeddings(self, prediction_interactions = None, embedding_dimension = 640,
                             replace_dataframe = True, return_normalisation_conststants = False, delimiter = '\t'):

        '''
            Reads in rnafm model generates embeddings for all rnas in rnas dataframe

            Inputs :
            embedding_dimension : Integer - Dimensions of rnafm embedding
            prediction_interactions : Pandas DataFrame - Dataframe with prediction information
            replace_dataframe : Bool - Replace existing rnas dataframe with one that contains rna Sequences and its respective normalised rnafm embedding
            return_normalisation_conststants : Bool - Returns normalisation constant if true
            delimiter : String - Delimiter for reading in Pandas rnafm DataFrame
        '''

        if type(prediction_interactions) != type(None):
            rna_list = list(prediction_interactions[self.rna_seq_name])
            replace_dataframe = False
        else:
            rna_list = self.rna_list

        rna_embeddings = np.zeros((len(rna_list), embedding_dimension))
        length_of_rna = [0 for _ in range(len(rna_list))]

        # Load RNA-FM model
        model, alphabet = fm.pretrained.rna_fm_t12('/src/checkpoints/')
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        # # For each RNA in rna list
        # for idx, RNA in tqdm(enumerate(rna_list)):
        #
        #     # 创建包含元组的列表
        #     if len(RNA) > 1022:
        #         RNA = RNA[:1022]
        #     RNA_tuple = [("RNA1", RNA)]
        #     #print(RNA_tuple)
        #     batch_labels, batch_strs, batch_tokens = batch_converter(RNA_tuple)
        #
        #     #print(batch_tokens)
        #
        #     with torch.no_grad():
        #         results = model(batch_tokens, repr_layers=[12])
        #     #print(results)
        #     token_embeddings = results["representations"][12]
        #     #print(token_embeddings)
        #     token_embeddings_np = token_embeddings.numpy()
        #     token_embeddings_np = token_embeddings_np[0,]
        #     token_embeddings_np_RNAlength = token_embeddings_np[1:-1, :]
        #     token_embeddings_np_RNAlength_mean = np.mean(token_embeddings_np_RNAlength, axis=0)
        #     #print(token_embeddings_np_RNAlength_mean)
        #     rna_embeddings[idx, :] = token_embeddings_np_RNAlength_mean
        #
        # self.rna_embeddings = rna_embeddings

        # For each RNA in rna list
        for idx, RNA in tqdm(enumerate(rna_list)):

            # 创建包含元组的列表
            if len(RNA) > 1022:
                # 指定每部分的长度
                chunk_size = 1022

                # 使用切片将字符串分成指定长度的部分
                chunks = [RNA[i:i + chunk_size] for i in range(0, len(RNA), chunk_size)]
                # token_embeddings_np_RNAlength = np.empty((1022,))
                # print(token_embeddings_np_RNAlength)
                # 对每个部分进行操作（示例：打印每个部分）
                for j, chunk in enumerate(chunks):
                    # print(j)
                    # print(chunk)
                    # 在这里进行你的操作
                    RNA_tuple = [("RNA1", chunk)]
                    # print(RNA_tuple)
                    batch_labels, batch_strs, batch_tokens = batch_converter(RNA_tuple)

                    # print(batch_tokens)

                    with torch.no_grad():
                        results = model(batch_tokens, repr_layers=[12])
                    # print(results)
                    token_embeddings_chunk = results["representations"][12]
                    # print(token_embeddings)
                    token_embeddings_chunk_np = token_embeddings_chunk.numpy()
                    token_embeddings_chunk_np = token_embeddings_chunk_np[0,]
                    token_embeddings_chunk_np_RNAlength = token_embeddings_chunk_np[1:-1, :]
                    token_embeddings_chunk_np_RNAlength_mean = np.mean(token_embeddings_chunk_np_RNAlength, axis=0)
                    if j == 0:
                        token_embeddings_np_RNAlength = token_embeddings_chunk_np_RNAlength_mean
                    else:
                        token_embeddings_np_RNAlength = np.vstack([token_embeddings_np_RNAlength, token_embeddings_chunk_np_RNAlength_mean])
                # print(token_embeddings_np_RNAlength_mean.shape)
                # print(token_embeddings_np_RNAlength_mean)
                token_embeddings_np_RNAlength_mean = np.mean(token_embeddings_np_RNAlength, axis=0)
                # print(token_embeddings_np_RNAlength_mean.shape)
                # print(token_embeddings_np_RNAlength_mean)
                rna_embeddings[idx, :] = token_embeddings_np_RNAlength_mean
                # print(token_embeddings_np_RNAlength_mean)

                # # 将一维数组转为二维数组（每个样本一个特征）
                # token_embeddings_np_RNAlength_mean_2d = token_embeddings_np_RNAlength_mean #.reshape(-1, 1)
                #
                # # 创建PCA模型并将数据压缩到640维
                # protein_dimension = 640
                # pca = PCA(n_components=protein_dimension)
                # compressed_token_embeddings_np_RNAlength_mean = pca.fit_transform(token_embeddings_np_RNAlength_mean_2d)
                #
                # # 获取PCA的投影矩阵
                # projection_matrix = pca.components_
                #
                # # 逆变换，将压缩后的数组映射回原始维度
                # token_embeddings_np_RNAlength_mean_2d = np.dot(compressed_token_embeddings_np_RNAlength_mean, projection_matrix.T)
                #
                # # 将二维数组还原为一维数组
                # token_embeddings_np_RNAlength_mean = token_embeddings_np_RNAlength_mean_2d.flatten()
                #
                #
                # rna_embeddings[idx, :] = token_embeddings_np_RNAlength_mean

                # # 指定目标长度
                # protein_length = 640
                #
                # # 计算降采样因子
                # downsampling_factor = len(token_embeddings_np_RNAlength_mean) // protein_length
                #
                # # 使用降采样将数组长度降低到目标长度
                # token_embeddings_np_RNAlength_mean = token_embeddings_np_RNAlength_mean[::downsampling_factor]
                # rna_embeddings[idx, :] = token_embeddings_np_RNAlength_mean

                # # 指定目标长度
                # protein_length = 640
                #
                # # 使用插值将数组长度插值到目标长度
                # token_embeddings_np_RNAlength_mean = np.interp(np.linspace(0, 1, protein_length), np.linspace(0, 1, len(token_embeddings_np_RNAlength_mean)), token_embeddings_np_RNAlength_mean)
                # rna_embeddings[idx, :] = token_embeddings_np_RNAlength_mean

            else:
                RNA_tuple = [("RNA1", RNA)]
                #print(RNA_tuple)
                batch_labels, batch_strs, batch_tokens = batch_converter(RNA_tuple)

                #print(batch_tokens)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[12])
                #print(results)
                token_embeddings = results["representations"][12]
                #print(token_embeddings)
                token_embeddings_np = token_embeddings.numpy()
                token_embeddings_np = token_embeddings_np[0,]
                token_embeddings_np_RNAlength = token_embeddings_np[1:-1, :]
                token_embeddings_np_RNAlength_mean = np.mean(token_embeddings_np_RNAlength, axis=0)
                # print(token_embeddings_np_RNAlength_mean)
                rna_embeddings[idx, :] = token_embeddings_np_RNAlength_mean

        self.rna_embeddings = rna_embeddings

        # Normalize embeddings - train data
        if type(prediction_interactions) == type(None):
            self.rna_embeddings = rna_embeddings
            self.mean_rna_embeddings = np.mean(rna_embeddings, axis=0)
            self.centered_rna_embeddings = rna_embeddings - self.mean_rna_embeddings
            self.centered_rna_embeddings_length = np.mean(np.sqrt(np.sum(self.centered_rna_embeddings * self.centered_rna_embeddings, axis=1)))
            #print(self.mean_rna_embeddings)

            #print(self.centered_rna_embeddings_length)
            #print(self.centered_rna_embeddings * self.centered_rna_embeddings)
            self.normalized_rna_embeddings = self.centered_rna_embeddings / np.expand_dims(
                self.centered_rna_embeddings_length, axis=-1)

        # Normalize for prediction data and return
        else:
            centered_rna_embeddings = rna_embeddings - self.mean_rna_embeddings
            normalized_rna_embeddings = centered_rna_embeddings / np.expand_dims(
                self.centered_rna_embeddings_length, axis=-1)
            rna_dataframe = pd.DataFrame([rna_list, normalized_rna_embeddings]).T
            rna_dataframe.columns = [self.rna_seq_name, self.rna_embedding_name]
            self.rna_dataframe = rna_dataframe
            return rna_dataframe
        # self.normalized_rna_embeddings = self.rna_embeddings
        # Replace proteins dataframe with
        if replace_dataframe:
            self.rna_dataframe = pd.DataFrame([rna_list, self.normalized_rna_embeddings]).T
            self.rna_dataframe.columns = [self.rna_seq_name, self.rna_embedding_name]

        if return_normalisation_conststants:
            return self.rna_embeddings, self.centered_rna_embeddings_length, self.normalized_rna_embeddings


    # Get Target Embeddings From ProtTrans
    def get_ProtTrans_embeddings(self, prediction_interactions=None, embedding_dimension=1024, replace_dataframe=True,
                                 return_normalisation_conststants=False, delimiter='\t'):
        '''
            Reads in ProtVec model generates embeddings for all proteins in proteins dataframe

            Inputs :
            embedding_dimension : Integer - Dimensions of ProtVec embedding
            prediction_interactions : Pandas DataFrame - Dataframe with prediction information
            replace_dataframe : Bool - Replace existing proteins dataframe with one that contains AA Sequences and its respective normalised ProtVec embedding
            return_normalisation_conststants : Bool - Returns normalisation constant if true
            delimiter : String - Delimiter for reading in Pandas ProtVec DataFrame
        '''

        if type(prediction_interactions) != type(None):
            protein_list = list(prediction_interactions[self.protein_seq_name])
            replace_dataframe = False
        else:
            protein_list = self.protein_list

        protein_embeddings = np.zeros((len(protein_list), embedding_dimension))
        length_of_protein = [0 for _ in range(len(protein_list))]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, legacy=True)

        # Load the model
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

        # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
        model.full() if device == 'cpu' else model.half()
        # model.to(device)

        # For each protein in protein list
        for idx, protein in tqdm(enumerate(protein_list)):
            # print(idx)
            # print(protein)
            # print(type(protein))
            protein = [protein]
            # print(type(protein))

            # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            #
            # # Load the tokenizer
            # tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
            #
            # # Load the model
            # model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
            #
            # # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
            # model.full() if device == 'cpu' else model.half()

            # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            protein = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein]


            # tokenize sequences and pad up to the longest sequence in the batch
            ids = tokenizer(protein, add_special_tokens=True, padding="longest")

            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            # generate embeddings
            with torch.no_grad():
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

            # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
            emb_0 = embedding_repr.last_hidden_state[0, :-1]  # shape (7 x 1024)
            # same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:8])
            # emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

            # if you want to derive a single representation (per-protein embedding) for the whole protein
            emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)
            protein_embeddings[idx, :] = emb_0_per_protein.cpu().numpy()



        self.protein_embeddings = protein_embeddings
        # self.mean_protein_embeddings = np.mean(protein_embeddings, axis=0)
        # Normalize embeddings - train data

        if type(prediction_interactions) == type(None):
            self.protein_embeddings = protein_embeddings
            self.mean_protein_embeddings = np.mean(protein_embeddings, axis = 0)
            self.centered_protein_embeddings = protein_embeddings - self.mean_protein_embeddings
            self.centered_protein_embeddings_length = np.mean(np.sqrt(np.sum(self.centered_protein_embeddings * self.centered_protein_embeddings, axis = 1)))
            self.normalized_protein_embeddings = self.centered_protein_embeddings / np.expand_dims(self.centered_protein_embeddings_length, axis = -1)

        # Normalize for prediction data and return
        else:
            centered_protein_embeddings = protein_embeddings - self.mean_protein_embeddings
            normalized_protein_embeddings = centered_protein_embeddings / np.expand_dims(self.centered_protein_embeddings_length, axis = -1)
            proteins_dataframe = pd.DataFrame([protein_list, normalized_protein_embeddings]).T
            proteins_dataframe.columns = [self.protein_seq_name, self.protein_embedding_name]
            self.proteins_dataframe = proteins_dataframe
            return proteins_dataframe

        # self.normalized_protein_embeddings = self.protein_embeddings
        # Replace proteins dataframe with
        if replace_dataframe:
            self.proteins_dataframe = pd.DataFrame([protein_list, self.normalized_protein_embeddings]).T
            self.proteins_dataframe.columns = [self.protein_seq_name, self.protein_embedding_name]

        if return_normalisation_conststants:
            return self.protein_embeddings, self.centered_protein_embeddings_length, self.normalized_protein_embeddings

################################################################
######      Predict given RNA and protein sequence        ######
################################################################

    def predict_RPI(self, model_dataset=None,
                          graphsage_path=None,
                          jobname=None,
                          test_dataframe=None,
                          rna_vector_length=None,
                          protein_vector_length=None,
                          rnas=None,
                          proteins=None,
                          embedding_type=None,
                          graphsage_embedding=1):

        '''
            Computes validation results

            Inputs :


            Outputs :


        '''

        if model_dataset == None:
            sys.exit("请输入正确的数据集！")

        if type(test_dataframe) == type(None):
            sys.exit("请输入测试集！")
        self.averaged_results = []

        for run in range(5):
            ### 获取graphsage的嵌入
            if graphsage_embedding == 1:

                graphsage_model_path = graphsage_path+'Run_' + str(run) + '/graphSage.pth'
                # print(graphsage_model_path)
                self.get_unseen_graphsage_embeddings(model_path=graphsage_model_path, model_dataset=model_dataset,
                                                              unseen_dataset=jobname, embedding_type=embedding_type,
                                                     rna_vector_length=rna_vector_length, protein_vector_length=protein_vector_length,
                                                     rnas=rnas, proteins=proteins, test_interactions=test_dataframe)

                self.normalized_protein_embeddings = self.graphsage_proteins_embeddings
                self.normalized_rna_embeddings = self.graphsage_rnas_embeddings

                X_0_test, X_1_test = self.dataframe_to_embed_array(
                    interactions_df=test_dataframe,
                    rna_list=self.rna_list,
                    protein_list=self.protein_list,
                    rna_embed_len=rna_vector_length+100,
                    normalized_protein_embeddings=self.normalized_protein_embeddings,
                    normalized_rna_embeddings=self.normalized_rna_embeddings,
                    include_true_label=False)

            else:
                test_rnas = rnas
                test_proteins = proteins
                # 将节点的embedding变成np矩阵
                rna_embeddings = test_rnas['normalized_embeddings']
                rna_array = np.zeros((len(test_rnas['normalized_embeddings']), rna_vector_length))
                # print(rna_embeddings)
                protein_embeddings = test_proteins['normalized_embeddings']
                protein_array = np.zeros((len(test_proteins['normalized_embeddings']), protein_vector_length))
                # print(protein_embeddings)
                # 使用 for 循环逐行赋值
                for i in range(len(test_rnas['normalized_embeddings'])):
                    # print(rna_embeddings.iloc[i].shape)
                    rna_array[i, :] = rna_embeddings.iloc[i]

                # 使用 for 循环逐行赋值
                for i in range(len(test_proteins['normalized_embeddings'])):
                    protein_array[i, :] = protein_embeddings.iloc[i]

                self.normalized_protein_embeddings = protein_array
                self.normalized_rna_embeddings = rna_array

                X_0_test, X_1_test = self.dataframe_to_embed_array(
                    interactions_df=test_dataframe,
                    rna_list=self.rna_list,
                    protein_list=self.protein_list,
                    rna_embed_len=rna_vector_length,
                    include_true_label=False)
            # print(X_0_test)
            # print(X_0_test.shape)
            # print(X_1_test)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_index = torch.cuda.current_device()
            # print(device_index)

            best_model_path = os.path.join(self.model_out_dir, 'Run_' + str(run), f"VecNN_5_fold_Benchmark_Dataset_{model_dataset}.pth")

            # 从检查点中加载模型状态字典
            #best_vecnet_model = torch.load(best_model_path,weights_only=False,map_location="cpu")
            best_vecnet_model = torch.load(best_model_path, weights_only=False)
            # if torch.cuda.device_count() > 1:
            #     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            #     best_vecnet_model = torch.nn.DataParallel(best_vecnet_model)
            best_vecnet_model.to(device)
            # print(best_vecnet_model)

            # 确保模型处于评估模式（不进行梯度计算）
            best_vecnet_model.eval()

            Y_test_predictions = best_vecnet_model(torch.tensor(X_0_test, dtype=torch.float32).to(device),
                                          torch.tensor(X_1_test, dtype=torch.float32).to(
                                              device)).detach().cpu().numpy()


            # print(Y_test_predictions)
            self.averaged_results.append(Y_test_predictions.flatten().tolist()[0])