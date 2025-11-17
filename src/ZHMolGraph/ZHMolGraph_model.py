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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from tqdm import tqdm
import os
import json
import time
import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
#import fm
import sys
import argparse
import pyhocon
import random
import csv

class VecNN(nn.Module):
    def __init__(self, target_embed_len=1024, rna_embed_len=640, graphsage_embedding=1):
        super(VecNN, self).__init__()
        # 根据输入向量的长度调整卷积层的大小
        if graphsage_embedding == 1:
            if target_embed_len == 1024:
                self.target_conv_len = 4488

            elif target_embed_len == 100:
                self.target_conv_len = 792

            elif target_embed_len == 49:
                self.target_conv_len = 584

            # print('self.target_conv_len\n')
            # print(self.target_conv_len)

            if rna_embed_len == 640:
                self.rna_conv_len = 2952
            elif rna_embed_len == 64:
                self.rna_conv_len = 648

        if graphsage_embedding == 0:
            if target_embed_len == 1024:
                self.target_conv_len = 4088

            elif target_embed_len == 100:
                self.target_conv_len = 392

            elif target_embed_len == 49:
                self.target_conv_len = 184

            # print('self.target_conv_len\n')
            # print(self.target_conv_len)
            if rna_embed_len == 640:
                self.rna_conv_len = 2552
            elif rna_embed_len == 64:
                self.rna_conv_len = 248


        # 添加一维卷积层
        self.conv1d_target = nn.Conv1d(1, 8, kernel_size=3)
        self.conv1d_rna = nn.Conv1d(1, 8, kernel_size=3)
        # self.relu1 = nn.ReLU()
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.target_layer = nn.Sequential(
            nn.ReLU(),
            # nn.Linear(8176, 2048),
            # nn.Linear(4488, 2048),
            # nn.Linear(792, 2048),
            nn.Linear(self.target_conv_len, 2048),
            nn.ReLU(),
        )

        self.rna_layer = nn.Sequential(
            nn.ReLU(),
            # nn.Linear(5104, 2048),
            # nn.Linear(512, 2048),
            # nn.Linear(2952, 2048),
            # nn.Linear(648, 2048),
            nn.Linear(self.rna_conv_len, 2048),
            nn.ReLU(),
        )

        self.concat_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(4096, 512),
            nn.ReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, target_input, rnas_input):
        # 添加一维卷积操作
        # print(target_input.unsqueeze(1).shape)
        # print(target_input.unsqueeze(1))
        target_input = self.conv1d_target(target_input.unsqueeze(1))
        rnas_input = self.conv1d_rna(rnas_input.unsqueeze(1))
        # print(target_input)
        # print(rnas_input)
        # print(target_input.shape)
        # print(rnas_input.shape)
        # target_input = self.relu1(target_input)
        # rnas_input = self.relu1(rnas_input)
        # print(target_input)
        # print(rnas_input)
        # print(target_input.shape)
        # print(rnas_input.shape)
        target_input = self.pool(target_input)
        rnas_input = self.pool(rnas_input)
        target_input = self.flatten(target_input)
        rnas_input = self.flatten(rnas_input)
        # print(target_input)
        # print(rnas_input)
        # print(target_input.shape)
        # print(rnas_input.shape)

        target_output = self.target_layer(target_input)
        rnas_output = self.rna_layer(rnas_input)

        combined = torch.cat((target_output, rnas_output), dim=1)
        x = self.concat_layer(combined)
        output = self.output_layer(x)

        return output


