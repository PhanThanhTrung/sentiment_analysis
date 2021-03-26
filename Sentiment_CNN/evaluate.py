import glob
import os

import gensim.models.keyedvectors as word2vec
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import load_model
from tqdm import tqdm

model_path = './models.h5'
model_embedding_path = './word_embedding.model'
data_path = '/Users/hit.fluoxetine/Dataset/nlp/data_test/'
model_embedding = word2vec.KeyedVectors.load(model_embedding_path)
model = load_model(model_path)
word_labels = []
max_seq = 200
embedding_size = 128

for word in model_embedding.vocab.keys():
    word_labels.append(word)


def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    lencmt = len(words)

    for i in range(max_seq):
        indexword = i % lencmt
        if (max_seq - i < lencmt):
            break
        if(words[indexword] in word_labels):
            matrix[i] = model_embedding[words[indexword]]
    matrix = np.array(matrix)
    return matrix


def readdata(path):
    all_data = []
    all_label = []
    list_file = glob.glob(path+"/*/*.txt")
    for elem in list_file:
        with open(elem, 'r') as f:
            data = f.read()
            label = elem.split('/')[-2]
            all_data.append(data)
            all_label.append(label)
    return all_data, all_label


if __name__ == '__main__':
    test_cmt, test_label = readdata(data_path+'test/')
    test_data = []
    label_data = []

    for x in tqdm(test_cmt, desc='testing set'):
        test_data.append(comment_embedding(x))
    test_data = np.array(test_data)
    test_data = np.expand_dims(test_data, axis=-1)
    for y in tqdm(test_label):
        if y == 'pos':
            label_data.append(0)
        else:
            label_data.append(1)
    test_y = np.array(label_data)
    y_predict = model.predict(test_data)
    top_classes = np.argmax(y_predict, axis=1)
    nums_samples = test_y.shape[0]
    wrong_prediction = np.sum(np.abs(top_classes-test_y))
    true_prediction = nums_samples - wrong_prediction
    print("accuracy on test set: {}".format(true_prediction/nums_samples))
