@echo off 

cd  U:\Dissertation\py

start

python dl_training.py rcnn industry
python dl_training.py rcnn region
python dl_training.py rcnn topics
python dl_training.py rnn industry
python dl_training.py rnn region
python dl_training.py rnn topics
python dl_training.py fasttext industry
python dl_training.py fasttext region
python dl_training.py fasttext topics
python dl_training.py cnn industry
python dl_training.py cnn region
python dl_training.py cnn topics
python tr_training.py decisiontree
python tr_training.py knn
python tr_training.py logistic
python tr_training.py svm