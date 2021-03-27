import numpy as np
import pandas as pd 
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

text = 'Đồ ăn ở đây rất ngon.'
model_path = './models.h5'
tokenizer_path = './tokenizer.pickle'

if __name__=='__main__':
    model= load_model(model_path)
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    X= tokenizer.texts_to_sequences(text)
    X = pad_sequences(X)
    result = model.predict(X.T)
    result = result.squeeze()
    top_class = np.argmax(result)
    if top_class ==0:
        type_cmt = 'Positive'
    else:
        type_cmt = 'Negative'
    print("Label predict: {}, Sore: {}".format(type_cmt,result[top_class]))