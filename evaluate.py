import logging
import os
import random
from collections import Counter

import glob
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torchtext.legacy.data import (BucketIterator, Field, LabelField,
                                   TabularDataset)
from transformers import AutoTokenizer
from phobert_lstm import phobert_lstm
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

if __name__ == '__main__':
    hidden_dim = 256
    num_classes = 2
    n_layers = 2
    bidirectional = True
    dropout = 0.25
    state_dict_path = './models/17_17_0.30.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phobert_path = "vinai/phobert-base"
    source_folder = '/root/dataset/sentiment_analysis/'
    batch_size = 8
    
    tokenizer = AutoTokenizer.from_pretrained(phobert_path, use_fast=False)
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
    
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
    train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='validation.csv', test='test.csv',
                                           format='CSV', fields=fields, skip_header=True)
    

    train_generator, val_generator, test_generator = BucketIterator.splits(
            (train, valid, test), 
            batch_size = batch_size, 
            device = device, sort = False)
    
    
    criterion = nn.CrossEntropyLoss()
    
    criterion = criterion.to(device)
    all_statedict_path = glob.glob('/root/logs/*.pth')
    for state_dict_path in all_statedict_path:
        print(state_dict_path)
        epoch_loss = 0
        epoch_acc = 0
        model = phobert_lstm(phobert_path=phobert_path,
                             state_dict_path = state_dict_path,
                             hidden_dim = hidden_dim,
                             num_classes = num_classes,
                             n_layers= n_layers,
                             bidirectional= bidirectional,
                             dropout = dropout,
                             device = device)
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(test_generator, desc = "Evaluating"):
                predictions = model(batch.data).squeeze(1)
                loss = criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
        print('Evaluating Accuracy ', epoch_acc/len(test_generator))
        print('Evaluating Loss ', epoch_loss/len(test_generator))
    

