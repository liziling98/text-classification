import torch
import torch.nn as nn
import csv
import itertools
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import sys
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(CNNNet, self).__init__()
        embedding_dim = 300
        
        max_features = len(embedding_matrix)
        self.embedding = nn.Embedding(max_features, embedding_dim)
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True
                
        content_dim = 150
        kernel_sizes = [1,2,3,4,5]
        linear_hidden_size = 128

        self.convs = nn.ModuleList([ nn.Sequential(
                        nn.Conv1d(in_channels = embedding_dim,
                            out_channels = content_dim,
                            kernel_size = k),
                            nn.BatchNorm1d(content_dim),
                            nn.ReLU(inplace=True),

                        nn.Conv1d(in_channels = content_dim,
                            out_channels = content_dim,
                            kernel_size = k),
                            nn.BatchNorm1d(content_dim),
                            nn.ReLU(inplace=True),
                            nn.MaxPool1d(kernel_size = (embedding_dim - k*2 + 2))
                ) for k in kernel_sizes])
        
        self.fc = nn.Sequential(
            nn.Linear(content_dim * len(kernel_sizes),linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size,num_classes)
        )
        
    def forward(self, x):
        x = self.embedding(x)  
#         x = x.unsqueeze(1)  
        x = [conv(x.permute(0,2,1)) for conv in self.convs]
        x = torch.cat(x, 1)
        reshaped = x.view(x.size(0), -1)
        logits = self.fc(reshaped)  
        
        return logits

class FastNet(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(FastNet, self).__init__() 
        embedding_dim = 300
        max_features = len(embedding_matrix)
        
        self.embedding = nn.Embedding(max_features, embedding_dim)
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True

        self.pre = nn.Sequential(
            nn.Linear(300, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(150, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        self.drop = nn.Dropout()
        
    def forward(self, x):
        x = self.embedding(x) #256 150 300
        content = self.pre(x)
        content = self.drop(content)
        avg = torch.mean(content,dim=1)
        output = self.fc(avg)  
        return output

class RCNNnet(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(RCNNnet, self).__init__()
        
        max_features = len(embedding_matrix)
        hidden_size = 64
        embedding_dim = 300
        sentence_dim = 150
        kernel_sizes = [2,3,4,5]
        linear_hidden_size = 128
        
        self.embedding = nn.Embedding(max_features, embedding_dim)
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = True
        
        self.lstm =nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = 2, bidirectional = True)

        self.convs = nn.ModuleList(nn.Sequential(
            nn.Conv1d(in_channels = hidden_size * 2 + embedding_dim,
                      out_channels = sentence_dim,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(sentence_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels = sentence_dim,
                      out_channels = sentence_dim,
                      kernel_size =  kernel_size),
            nn.BatchNorm1d(sentence_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size = (sentence_dim - kernel_size * 2 + 2))
        ) for kernel_size in kernel_sizes)
        
        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * sentence_dim, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size, num_classes)
        )
        
    def forward(self, x):
        content = self.embedding(x)
        # lstm
        content_out, _ = self.lstm(content)
        content_out = torch.cat((content_out.permute(1,2,0),content.permute(1,2,0)),dim=1).permute(2,1,0)
        # conv
        conv_out = [conv(content_out) for conv in self.convs]
        conv_out = torch.cat(conv_out, dim = 1)
        # fully connected
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits

class RNNnet(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(RNNnet, self).__init__() 
        ## Embedding Layer, Add parameter
        max_features = len(embedding_matrix)
        embed_size = 300
        linear_hidden_size = 64     

        self.embedding = nn.Embedding(max_features, embed_size)
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = False
#         self.embedding_dropout = nn.Dropout2d(0.5)
        
        self.lstm = nn.LSTM(300, 128, num_layers = 2, bidirectional = True)   
        self.fc = nn.Sequential(
            nn.Linear(256, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size, len(industry_codes))
        )
        
    def forward(self, x):
        h_embedding = self.embedding(x)      
        h_lstm, _ = self.lstm(h_embedding)
        max_pool, _ = torch.max(h_lstm, 1)
        output = self.fc(max_pool)
        return output