import numpy as np
import pandas as pd 
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import glob
model_path = './models.h5'
tokenizer_path = './tokenizer.pickle'
data_path= '/Users/hit.fluoxetine/Dataset/nlp/data_test/'
def readdata(path):
    all_data = []
    all_label =[]
    list_file = glob.glob(path+"/*/*/*.txt")
    for elem in list_file:
        with open(elem,'r') as f:
            data = f.read()
            label = elem.split('/')[-2]
            all_data.append(data)
            if label =='neg':
                all_label.append(1)
            else:
                all_label.append(0)
    return all_data, all_label


if __name__ == '__main__':
    model = load_model(model_path)
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    reviews, labels = readdata(data_path)
    labels= np.array(labels)

    X_test= tokenizer.texts_to_sequences(reviews)
    X_test = pad_sequences(X_test)
    result = model.predict(X_test)
    result = result.squeeze()
    top_classes = np.argmax(result, axis= 1)
    nums_samples = labels.shape[0]
    wrong_prediction = np.sum(np.abs(top_classes-labels))
    true_prediction = nums_samples - wrong_prediction
    print("accuracy on test set: {}".format(true_prediction/nums_samples))