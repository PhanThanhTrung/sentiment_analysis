import logging
import os
import random
from collections import Counter
from re import S
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from phobert_lstm import phobert_lstm
from torchtext.legacy.data import (BucketIterator, Field, LabelField,
                                   TabularDataset)

from transformers import AutoTokenizer
from clearml import Task

task = Task.init(project_name='sentiment_analysis',
                 task_name='29042021_LSTM_bidirectional_2layer_phoBERT')

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
    source_file = '/home/miles/HIT/sentiment_analysis/test.txt'
    state_dict_path = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phobert_path = "vinai/phobert-base"
    SOURCE_FOLDER = '/root/dataset/sentiment_analysis/'
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    log_dir = '/root/logs/'

    model = phobert_lstm(phobert_path,
                         state_dict_path,
                         hidden_dim,
                         num_classes,
                         n_layers,
                         bidirectional,
                         dropout,
                         device)
    
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
    
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tensorboardX.SummaryWriter()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    global_count = 0
    for epoch in range(NUM_EPOCHS):
        logger.info('Training on epoch {}'.format(epoch+1))
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for batch in tqdm.tqdm(train_generator, desc = 'Training'):
            optimizer.zero_grad()
            predictions = model(batch.data).squeeze(1)
            loss = criterion(predictions, batch.label)
            loss.backward()
            optimizer.step()
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            
        writer.add_scalar('Training_Accuracy', epoch_acc/len(train_generator), global_count)
        writer.add_scalar('Training_Loss', epoch_loss/len(train_generator), global_count)
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(val_generator, desc = "Validation"):
                predictions = model(batch.data).squeeze(1)
                
                loss = criterion(predictions, batch.label)
                
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
        writer.add_scalar('Validation_Accuracy', epoch_acc/len(val_generator), global_count)
        writer.add_scalar('Validation_Loss', epoch_loss/len(val_generator), global_count)
        

        name = '{}_{}_{:.2f}.pth'.format(epoch, global_count, epoch_loss/len(val_generator))
        path_model_state_dict = os.path.join(
            log_dir, name)
        torch.save(model.state_dict(), path_model_state_dict)
        global_count+=1
