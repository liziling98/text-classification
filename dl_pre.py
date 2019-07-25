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

def load_data(filename):
    container = {}
    # i = 0
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            temp = {}
            temp['itemid'] = row['itemid']
            temp['codes'] = row['codes']
            temp['text'] = row['text']
            container[row['itemid']] = temp
            # i += 1
            # if i > 1000:
            #     break
        csvfile.close()
    return container

def text_clean(text, stemmer, stop):
    #tokenisation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    # stemming
    for i in range(len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])
    # remove stop words
    test_remove_ = [i for i in tokens if i not in stop]
    return test_remove_

def code_clean(code):
    temp = code.replace("[", "").replace("]","").replace("\'", "").replace(" ", "")
    return temp.split(",")

def check_code_exist(lis, codes_lis, temp):
    for code in codes_lis:
        if code in lis:
            temp.append(1)
        else:
            temp.append(0)

def preprocessing(data_file, label_file):
    nltk.download('stopwords')
    stop = stopwords.words('english')
    stemmer = nltk.PorterStemmer()

    # get codes and text after preprocessing
    info = load_data(data_file)
    for itemid, content in info.items():
        content['codes'] = code_clean(content['codes'])
        content['text'] = text_clean(content['text'], stemmer, stop)

    # get labels
    f = open(label_file, "r")
    industry = f.read()
    f.close()
    labels = code_clean(industry)

    for idx, content in info.items():
        all_lis = []
        check_code_exist(content['codes'], labels, all_lis)
        content['one_hot_code'] = all_lis

    # constructing dataset for trainging and testing
    train = {}
    x_train = []
    y_train = []
    for itemid, content in info.items():
        train[itemid] = content
        x_train.append(content['text'])
        y_train.append(content['one_hot_code'])
    y_train = np.array(y_train).T
    
    return x_train, y_train, labels

def clean(code):
    temp = code.replace("[", "").replace("]","").replace("\r", "").replace("\n", "").replace("'", '')
    science_type = temp.split()
    output = [float(s) for s in science_type]
    output = np.asarray(output, dtype='float32')
    return output

def load_embedding(filename):
    word_embedding = {}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            temp = {}
            temp['index'] = row['index']
            temp['embedding'] = clean(row['embedding'])
            word_embedding[row['word']] = temp
        csvfile.close()
    return word_embedding
    
def embedding(traing_data, filename):
    word_embedding = load_embedding(filename)

    corpus = [sentence for sentence in traing_data]
    vocabulary = set(itertools.chain.from_iterable(corpus))

    embedding_matrix = np.zeros((len(vocabulary) + 1, 300))
    word2idx = {}
    i = 0
    for w in list(vocabulary):
        if w in word_embedding.keys():
            embedding_matrix[i] = word_embedding[w]['embedding']
        else:
            embedding_matrix[i] = np.zeros(300)
        word2idx[w] = i
        i += 1
    return word2idx, embedding_matrix

def sentence_split(training, lence, word2idx):
    text_set = []
    for i in range(len(training)):
        diff = lence - len(training[i])
        temp = []
        while diff >= 1:
            temp.append(0)
            diff -= 1
        for word in training[i][:lence]:
            temp.append(word2idx[word]) if word in word2idx else 0
        text_set.append(temp)
    return np.array(text_set)

def split_dataset(text_set, y_train, batch_size):
    train_size = int(0.7 * len(text_set))
    test_size = len(text_set) - train_size

    x_ = torch.tensor(text_set, dtype=torch.long).cuda()
    y_ = torch.tensor(y_train.T, dtype=torch.float32).cuda()
    dataset = TensorDataset(x_, y_)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, validloader

def test(model, test_loader):
    A_lis = []
    B_lis = []
    for item in test_loader:
        output = model(item[0])
        two_class = torch.sign(output, out=None)
        preds_tensor = torch.clamp(two_class, 0, 1, out=None)
        y_pred = np.squeeze(preds_tensor.cpu().detach().numpy())
        true_labels = np.squeeze(item[1].cpu().detach().numpy())
        A_lis.append(list(itertools.chain.from_iterable(true_labels)))
        B_lis.append(list(itertools.chain.from_iterable(y_pred)))
    pred_lis = list(itertools.chain.from_iterable(A_lis))
    label_lis = list(itertools.chain.from_iterable(B_lis))
    print('macro F1: {}'.format(f1_score(label_lis, pred_lis, average = 'macro')), "\n")
    print('macro precision score: {}'.format(precision_score(label_lis, pred_lis, average = 'micro')), "\n")
    print('macro recall score: {}'.format(recall_score(label_lis, pred_lis, average = 'micro')), "\n")
    print('binary F1: {}'.format(f1_score(label_lis, pred_lis, average = 'binary')), "\n")
    print('binary precision score: {}'.format(precision_score(label_lis, pred_lis, average = 'binary')), "\n")
    print('binary recall score: {}'.format(recall_score(label_lis, pred_lis, average = 'binary')), "\n")