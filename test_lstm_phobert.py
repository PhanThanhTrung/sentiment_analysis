import torch
from transformers import AutoModel, AutoTokenizer
from CocCocTokenizer import PyTokenizer
import torch.nn as nn
import logging
from phobert_lstm import phobert_lstm
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from CocCocTokenizer import PyTokenizer
T = PyTokenizer(load_nontone_data=True)

def tokenize_and_cut(sentence: str, max_input_length:int= 256):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

def pre_process(sentence: str):
    sentence = sentence.lower() 
    sentence = sentence.strip('\n').strip(' ') 
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' 
    no_punct = ""
    for char in sentence:
        if char not in punctuations:
            no_punct = no_punct + char
    
    splitted = no_punct.split()
    normed_sentence = ' '.join(splitted)
    return normed_sentence

def prepair_input():
    pass

if __name__ == '__main__':
    LABELS = ['POSITVIVE', 'NEGATIVE']
    hidden_dim = 256
    num_classes = 2
    n_layers = 2
    bidirectional = True
    dropout = 0.25
    source_file = '/home/miles/HIT/sentiment_analysis/test.txt'
    state_dict_path = '/home/miles/Downloads/17_17_0.30.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phobert_path = "vinai/phobert-base"
    
    model = phobert_lstm(phobert_path=phobert_path,
                         state_dict_path = state_dict_path,
                         hidden_dim = hidden_dim,
                         num_classes = num_classes,
                         n_layers= n_layers,
                         bidirectional= bidirectional,
                         dropout = dropout,
                         device = device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(phobert_path, use_fast=False)

    all_data = []
    with open(source_file, 'r') as f:
        all_data = f.readlines()
    with torch.no_grad():
        for sentence in all_data: 
            sentence = ' '.join(T.word_tokenize(sentence, tokenize_option=0))
            tokens = tokenizer.tokenize(sentence)
            tokens = tokens[:256-2]
            indexed = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
            indexed_tensor = torch.LongTensor(indexed).to(device)
            indexed_tensor = indexed_tensor.unsqueeze(0)
            y_pred = model.forward(indexed_tensor)
            y_pred = y_pred.squeeze()
            softmax_pred = torch.softmax(y_pred, dim = 0)
            LABELS = ['NEGATIVE', 'POSITIVE']
            predict_class = LABELS[torch.argmax(softmax_pred)]
        