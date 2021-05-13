import logging
import os
import random
from collections import Counter

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX
import torch
#from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.legacy.data import (BucketIterator, Field, LabelField,
                                   TabularDataset)
from torchtext.vocab import Vocab
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def tokenize_and_cut(sentence: str, max_input_length:int= 256):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

def binary_accuracy(preds, y):
    """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.argmax(torch.softmax(preds, dim = 1), dim = 1)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

class PhoBERTLSTMSentiment(nn.Module):
    def __init__(self,
                 phobert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.phobert = phobert
        
        embedding_dim = phobert.config.to_dict()['hidden_size']
        
        self.rnn = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, text):
        with torch.no_grad():
            embedded = self.phobert(text)[0]
        
        packed_output, (hidden, cell) = self.rnn(embedded)
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        
        output = self.out(hidden)
        
        return output

if __name__ == '__main__':
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    SOURCE_FOLDER = '/root/dataset/sentiment_analysis/'
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LOG_ITER = 50
    log_dir = '/root/logs/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    model = PhoBERTLSTMSentiment(phobert,
                            HIDDEN_DIM,
                            OUTPUT_DIM,
                            N_LAYERS,
                            BIDIRECTIONAL,
                            DROPOUT)
    
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
    max_input_length = tokenizer.max_model_input_sizes['vinai/phobert-base']
    TEXT = Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)
    LABEL = LabelField(dtype = torch.long, use_vocab =False)
    fields = [('data', TEXT), ('label', LABEL)]
    train, valid, test = TabularDataset.splits(path=SOURCE_FOLDER, train='train.csv', validation='validation.csv', test='test.csv',
                                           format='CSV', fields=fields, skip_header=True)
    

    train_generator, val_generator, test_generator = BucketIterator.splits(
            (train, valid, test), 
            batch_size = BATCH_SIZE, 
            device = device, sort = False)
    
    
    criterion = nn.CrossEntropyLoss()
    
    criterion = criterion.to(device)
    all_statedict_path = glob.glob('/root/logs/*.pth')
    for state_dict_path in all_statedict_path:
        print(state_dict_path)
        epoch_loss = 0
        epoch_acc = 0
        state_dict = torch.load(state_dict_path)
        model = PhoBERTLSTMSentiment(phobert,
                                HIDDEN_DIM,
                                OUTPUT_DIM,
                                N_LAYERS,
                                BIDIRECTIONAL,
                                DROPOUT)
        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(test_generator, desc = "Validation"):
                predictions = model(batch.data).squeeze(1)
                
                loss = criterion(predictions, batch.label)
                
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
        print('Validation_Accuracy ', epoch_acc/len(test_generator))
        print('Validation_Loss ', epoch_loss/len(test_generator))
    

