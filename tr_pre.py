import csv
import itertools
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

def load_code(filename1, filename2, filename3):
    f = open(filename1, "r", encoding='utf-8')
    industry = f.read()
    f.close()
    industry_codes = code_clean(industry)

    f = open(filename2, "r", encoding='utf-8')
    region = f.read()
    f.close()
    region_codes = code_clean(region)

    f = open(filename3, "r", encoding='utf-8')
    topics = f.read()
    f.close()
    topic_codes = code_clean(topics)

    return industry_codes, region_codes, topic_codes

def check_code_exist(lis, codes_lis, temp):
    for code in codes_lis:
        if code in lis:
            temp.append(1)
        else:
            temp.append(0)

def data_preprocess(filename):
    info = {}
    # i = 0
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            temp = {}
            temp['itemid'] = row['itemid']
            temp['codes'] = row['codes']
            temp['title'] = row['title']
            temp['text'] = row['text']
            info[row['itemid']] = temp
            # i += 1
            # if i> 1000:
            #     break
        csvfile.close()

    stop = stopwords.words('english')
    stemmer = nltk.PorterStemmer()
    for itemid, content in info.items():
        content['text'] = text_clean(content['text'], stemmer, stop)
        content['codes'] = code_clean(content['codes'])

    filename1 = "C:\\Users\\li\\dissertation\\industry.txt"
    filename2 = "C:\\Users\\li\\dissertation\\region.txt"
    filename3 = "C:\\Users\\li\\dissertation\\topics.txt"
    industry_codes, region_codes, topic_codes = load_code(filename1, filename2, filename3)

    for idx, content in info.items():
        industry_lis = []
        region_lis = []
        topic_lis = []
        check_code_exist(content['codes'], industry_codes, industry_lis)
        check_code_exist(content['codes'], region_codes, region_lis)
        check_code_exist(content['codes'], topic_codes, topic_lis)
        content['industry'] = industry_lis
        content['region'] = region_lis
        content['topic'] = topic_lis

    corpus = []
    industry_label = []
    region_label = []
    topic_label = []

    for idx, content in info.items():
        str_ = ''
        for word in content['text']:
            str_ += word
            str_ += ' '
        str_ += content['title']
        corpus.append(str_)
        industry_label.append(content['industry'])
        region_label.append(content['region'])
        topic_label.append(content['topic'])
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    svd = TruncatedSVD(n_components=10, n_iter=2, random_state=42).fit(X)
    X = svd.fit_transform(X)

    industry_label = np.array(industry_label)
    region_label = np.array(region_label)
    topic_label = np.array(topic_label)

    return X, industry_label, topic_label, region_label

def evaluation(labels, predictions, standard):
    f1 = f1_score(labels, predictions, average = standard)
    precision = precision_score(labels, predictions, average = standard)
    recall = recall_score(labels, predictions, average = standard)
    return f1, precision, recall

def train_test1(train, test, model_industry, model_topics, model_region, train_label_in, train_label_to, train_label_re, test_label_in, test_label_to, test_label_re):
    model_in = model_industry.fit(train, train_label_in)
    model_to = model_topics.fit(train, train_label_to)
    model_re = model_region.fit(train, train_label_re)

    prediction_in = model_in.predict(test)
    true = list(itertools.chain.from_iterable(test_label_in))
    pred = list(itertools.chain.from_iterable(prediction_in))
    f1, p, r = evaluation(true, pred, 'macro')
    print("for industry class, macro F1, precision score, recall socre: ", f1, p, r, "\n")
    f1, p, r = evaluation(true, pred, 'binary')
    print("for industry class, binary F1, precision score, recall socre: ", f1, p, r, "\n")

    prediction_to = model_to.predict(test)
    true = list(itertools.chain.from_iterable(test_label_to))
    pred = list(itertools.chain.from_iterable(prediction_to))
    f1, p, r = evaluation(true, pred, 'macro')
    print("for topics class, macro F1, precision score, recall socre: ", f1, p, r, "\n")
    f1, p, r = evaluation(true, pred, 'binary')
    print("for topics class, binary F1, precision score, recall socre: ", f1, p, r, "\n")

    prediction_re = model_re.predict(test)
    true = list(itertools.chain.from_iterable(test_label_re))
    pred = list(itertools.chain.from_iterable(prediction_re))
    f1, p, r = evaluation(true, pred, 'macro')
    print("for region class, macro F1, precision score, recall socre: ", f1, p, r, "\n")
    f1, p, r = evaluation(true, pred, 'binary')
    print("for region class, binary F1, precision score, recall socre: ", f1, p, r, "\n")

def train_test2(train, test, model_industry, model_topics, model_region, train_label_in, train_label_to, train_label_re, test_label_in, test_label_to, test_label_re):
    pred_in = []
    for i in range(len(train_label_in.T)):
        if sum(train_label_in.T[i]) != 0:
            model = model_industry.fit(train, train_label_in.T[i])
            pred = model.predict(test)
            pred_in.append(pred)
        else:
            temp = np.zeros(len(test))
            pred_in.append(temp)
    pred_in = np.array(pred_in).T
    pred = list(itertools.chain.from_iterable(pred_in))
    true = list(itertools.chain.from_iterable(test_label_in))
    f1, p, r = evaluation(true, pred, 'macro')
    print("for industry class, macro F1, precision score, recall socre: ", f1, p, r, "\n")
    f1, p, r = evaluation(true, pred, 'binary')
    print("for industry class, binary F1, precision score, recall socre: ", f1, p, r, "\n")

    pred_to = []
    for i in range(len(train_label_to.T)):
        if sum(train_label_to.T[i]) != 0:
            model = model_topics.fit(train, train_label_to.T[i])
            pred = model.predict(test)
            pred_to.append(pred)
        else:
            temp = np.zeros(len(test))
            pred_to.append(temp)
    pred_to = np.array(pred_to).T
    pred = list(itertools.chain.from_iterable(pred_to))
    true = list(itertools.chain.from_iterable(test_label_to))
    f1, p, r = evaluation(true, pred, 'macro')
    print("for topics class, macro F1, precision score, recall socre: ", f1, p, r, "\n")
    f1, p, r = evaluation(true, pred, 'binary')
    print("for topics class, binary F1, precision score, recall socre: ", f1, p, r, "\n")

    pred_re = []
    for i in range(len(train_label_re.T)):
        if sum(train_label_re.T[i]) != 0:
            model = model_region.fit(train, train_label_re.T[i])
            pred = model.predict(test)
            pred_re.append(pred)
        else:
            temp = np.zeros(len(test))
            pred_re.append(temp)
    pred_re = np.array(pred_re).T
    pred = list(itertools.chain.from_iterable(pred_re))
    true = list(itertools.chain.from_iterable(test_label_re))
    f1, p, r = evaluation(true, pred, 'macro')
    print("for region class, macro F1, precision score, recall socre: ", f1, p, r, "\n")
    f1, p, r = evaluation(true, pred, 'binary')
    print("for region class, binary F1, precision score, recall socre: ", f1, p, r, "\n")

