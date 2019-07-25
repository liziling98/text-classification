from dl_pre import load_data, text_clean, code_clean, check_code_exist, preprocessing, clean, load_embedding, embedding, sentence_split, split_dataset, test
from dl_models import CNNNet, FastNet, RCNNnet, RNNnet

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
from tqdm import tqdm,trange

model_ = str(sys.argv[1])
main_class = str(sys.argv[2])

if main_class == 'industry':
    label_file = "U:\\Dissertation\\industry.txt"
elif main_class == 'region':
    label_file = "U:\\Dissertation\\industry.txt"
elif main_class == 'topics':
    label_file = "U:\\Dissertation\\topics.txt"

data_file = 'U:\\Dissertation\\training_set.csv'
word_embedding_file = 'U:\\Dissertation\\word_embedding.csv'

x_train, y_train, labels = preprocessing(data_file, label_file)
    
word2idx, embedding_matrix = embedding(x_train, word_embedding_file)

text_set = sentence_split(x_train, 150, word2idx)

trainloader, validloader = split_dataset(text_set, y_train, batch_size=256)

if model_ == 'fasttext':
    model = FastNet(len(labels), embedding_matrix).cuda()
elif model_ == 'cnn':
    model = CNNNet(len(labels), embedding_matrix).cuda()
elif model_ == 'rnn':
    model = RNNnet(len(labels), embedding_matrix).cuda()
elif model_ == 'rcnn':
    model = RCNNnet(len(labels), embedding_matrix).cuda()

print("\n",model_," is training in",main_class,"class","\n")

txt_name = model_ + '_' + main_class

def training_iter(model, train_loader, valid_loader):
    record = []
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    learning_rate = 0.01
    for epoch in trange(1, 11, desc='1st loop'):
        learning_rate = learning_rate / (1 + 0.01 * epoch)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_loss = [], []
        ## training part
        model.train()
        for _,(data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        record.append(sum(train_loss))
    print('training loss:', record, '\n')
    return model

trained_model = training_iter(model, trainloader, validloader)

test(trained_model, validloader)
