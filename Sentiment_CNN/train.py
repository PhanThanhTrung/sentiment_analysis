# -*- coding: utf-8 -*-
import glob
import os

import gensim.models.keyedvectors as word2vec
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

data_path= '/Users/hit.fluoxetine/Dataset/nlp/data_train/'

model_embedding = word2vec.KeyedVectors.load('./word_embedding.model')

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
    all_label =[]
    list_file = glob.glob(path+"/*/*.txt")
    for elem in list_file:
        with open(elem,'r') as f:
            data = f.read()
            label = elem.split('/')[-2]
            all_data.append(data)
            all_label.append(label)
    return all_data, all_label

if __name__ == '__main__':
    
    train_cmt, train_label = readdata(data_path+'train/')
    train_data = []
    label_data = []

    for x in tqdm(train_cmt, desc ='training set'):
        train_data.append(comment_embedding(x))
    train_data = np.array(train_data)

    for y in tqdm(train_label):
        
        if y == 'pos':
            label_data.append([1,0])    
        else:
            label_data.append([0,1])

    test_cmt, test_label = readdata(data_path+'test/')
    test_data = []
    test_set = []

    for x in tqdm(test_cmt, desc = 'validation set'):
        test_data.append(comment_embedding(x))
    test_data = np.array(test_data)

    for y in tqdm(test_label):
        
        if y == 'pos':
            test_set.append([1,0])    
        else:
            test_set.append([0,1])

    sequence_length = 200
    embedding_size = 128
    num_classes = 3
    filter_sizes = 3
    num_filters = 150
    epochs = 50
    batch_size = 30
    learning_rate = 0.01
    dropout_rate = 0.5

    x_train = train_data.reshape(
        train_data.shape[0], sequence_length, embedding_size, 1).astype('float32')
    y_train = np.array(label_data)

    x_val = test_data.reshape(
        test_data.shape[0], sequence_length, embedding_size, 1).astype('float32')
    y_val = np.array(test_set)

    # Define model
    model = keras.Sequential()
    model.add(layers.Convolution2D(num_filters, (filter_sizes, embedding_size),
                                padding='valid',
                                input_shape=(sequence_length, embedding_size, 1), activation='relu'))
    model.add(layers.MaxPooling2D(198,1))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    # Train model
    tensorboard_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)
    adam = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, 
            batch_size = batch_size, 
            verbose=1, epochs=epochs, 
            callbacks=[tensorboard_callback], 
            validation_data=(x_val, y_val))
    model.save('models.h5')