import os
import glob
import gensim.models.keyedvectors as word2vec
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from tensorflow import keras
from tensorflow.keras import layers

path= '/Users/hit.fluoxetine/Dataset/nlp/data_train/'

def readdata(path):
    all_data = []
    all_label =[]
    list_file = glob.glob(path+"/*/*/*.txt")
    for elem in list_file:
        with open(elem,'r') as f:
            data = f.read()
            label = elem.split('/')[-2]
            all_data.append(data)
            all_label.append(label)
    return all_data, all_label
reviews, labels = readdata(path)

input_gensim = []
for review in reviews:
    input_gensim.append(review.split())

model = Word2Vec(input_gensim, size=128, window=5,
                 min_count=0, workers=4, sg=1)
model.wv.save("word_embedding.model")
