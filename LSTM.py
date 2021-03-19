import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import glob 
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
            if label =='neg':
                all_label.append([0,1])
            else:
                all_label.append([1,0])
    return all_data, all_label
reviews, labels = readdata(path)

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(reviews)
X = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(X)
Y = np.array(labels, np.float32)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128,activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

tensorboard_callback = keras.callbacks.TensorBoard('./logs_LSTM', update_freq=1)
adam = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])
epochs = 100
batch_size = 32
model.fit(X_train, Y_train, 
            batch_size = batch_size, 
            verbose=1, epochs=epochs, 
            callbacks=[tensorboard_callback], 
            validation_data=(X_test, Y_test))


