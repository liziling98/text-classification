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
from sklearn.decomposition import PCA
import sys
from sklearn.linear_model import LogisticRegression

from tr_pre import text_clean, code_clean, load_code, check_code_exist, data_preprocess, train_test1, train_test2

choice = str(sys.argv[1])
train_file = "C:\\Users\\li\\dissertation\\training_set.csv"
test_file = "C:\\Users\\li\\dissertation\\testing_set.csv"

train_set, industry_train, topic_train, region_train = data_preprocess(train_file)
test_set, industry_test, topic_test, region_test = data_preprocess(test_file)

if choice == 'decisiontree':
    print("Decision Tree is processing......", "\n")
    model_in = DecisionTreeClassifier(criterion = 'gini', max_depth= 20, min_samples_leaf= 2)
    model_to = DecisionTreeClassifier(criterion= 'gini', max_depth= 20, min_samples_leaf= 1)
    model_re = DecisionTreeClassifier(criterion= 'gini', max_depth= 25, min_samples_leaf= 2)
    train_test1(train_set, test_set, model_in, model_to, model_re, industry_train, topic_train, region_train, industry_test, topic_test, region_test)
elif choice == 'knn':
    print("KNN is processing......", "\n")
    model_in = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size= 20, n_neighbors= 5, p= 2, weights= 'uniform')
    model_to = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size = 20, n_neighbors= 3, p= 2, weights = 'uniform')
    model_re = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size= 20, n_neighbors= 3, p= 2, weights= 'distance')
    train_test1(train_set, test_set, model_in, model_to, model_re, industry_train, topic_train, region_train, industry_test, topic_test, region_test)
elif choice == 'logistic':
    print("Logistic regression is processing......", "\n")
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    train_test2(train_set, test_set, clf, clf, clf, industry_train, topic_train, region_train, industry_test, topic_test, region_test)
elif choice == 'svm':
    print("SVM is processing......", "\n")
    svc = svm.LinearSVC(class_weight = 'balanced')
    train_test2(train_set, test_set, svc, svc, svc, industry_train, topic_train, region_train, industry_test, topic_test, region_test)