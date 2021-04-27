import glob
import os

import tqdm
from CocCocTokenizer import PyTokenizer
from gensim.models import KeyedVectors, Word2Vec
import string

T = PyTokenizer(load_nontone_data=True)

def read_data(path):
    all_data = []
    list_file = glob.glob(path+"/*/*/*/*.txt")
    for elem in tqdm.tqdm(list_file, desc = 'loading all data'):
        with open(elem, 'r') as f:
            data = f.read()
            label = elem.split('/')[-2]
            all_data.append(
                {
                    'file_name': elem, 
                    'data': data, 
                    'label': label
                }
            )
    return all_data

if __name__ == '__main__':
    model = Word2Vec.load('/Users/hit.fluoxetine/HIT/repo/sentiment_analysis/data_processing/models/word_embedding_5.model') 
    corpus_path = '/Users/hit.fluoxetine/Downloads/corpus-full-0.2.txt'
    batch_size = 100000
    index = 0
    with open(corpus_path, 'r') as file:
        cur_batch =[]
        for line in tqdm.tqdm(file, desc = "Vietnamese Corpus"):
            if len(cur_batch) < batch_size:
                striped_data = line.strip('\n').strip(' ')
                raw_data= striped_data.replace('_', ' ')
                table = str.maketrans('', '', string.punctuation)  
                splitted = [w.translate(table) for w in raw_data]
                raw_data=  ''.join(splitted)
                raw_data= raw_data.lower()
                raw_data= raw_data.split()
                raw_data= ' '.join(raw_data)
                tokenized_sentence = T.word_tokenize(raw_data, tokenize_option=0)
                cur_batch.append(tokenized_sentence)
            if len(cur_batch) == batch_size:
                model.build_vocab(cur_batch, update=True)
                model.train(cur_batch, total_examples=model.corpus_count, epochs=model.epochs)  
                cur_batch = []
                index +=1
                if index % 100 == 0:
                    model.save('./models/word_embedding_{}.model'.format(index//100))
    
    model.save('./models/word_embedding_final.model')