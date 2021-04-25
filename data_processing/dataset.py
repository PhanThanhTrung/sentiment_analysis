import glob
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class DatasetLoader(Dataset):
    def __init__(self,
                 folder_path: str = None,
                 embedding_model_path: str = None
                 ) -> None:

        self.file_path_and_type = []
        self._load_dataset(folder_path)
        self.vocab_size = 0
        self.word2vec_model = None

        if embedding_model_path is not None and os.path.exists(embedding_model_path):
            self.word2vec_model = KeyedVectors.load_word2vec_format(embedding_model_path, binary=True)
            word_labels = []
            for key in self.word2vec_model.vocab.keys():
                word_labels.append(key)
            self.vocab = word_labels

    def __len__(self):
        return len(self.file_path_and_type)

    def __getitem__(self, idx):
        file_path, comment_type = self.file_path_and_type[idx]
        with open(file_path, 'r') as f:
            comment_content = f.read()
        comment_content = comment_content.strip(' ')
        comment_content_splitted = comment_content.split(' ')
        comment_embedding= np.zeros((len(comment_content_splitted), 400))
        
        for index, word in enumerate(comment_content_splitted):
            if word not in [' ', '', None] and word in self.vocab:
                embed_value= self.word2vec_model[word]
                shape = embed_value.shape
                comment_embedding[index] = embed_value
            

        comment_embedding= torch.tensor(comment_embedding, dtype = torch.float32)
        comment_type = torch.tensor(comment_type, dtype = torch.int64)
        return comment_embedding, comment_type

    def _load_dataset(self, folder_path: str = None) -> None:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(
                "No such file or directory: {}".format(folder_path))
        if not os.path.isdir(folder_path):
            raise FileNotFoundError("Your directory is not a folder!")
        max_length = 0
        all_data = glob.glob('{}/*/*.txt'.format(folder_path))
        for cur_file_path in all_data:
            cur_label = cur_file_path.split('/')[-2]
            cur_file_type = 0
            if cur_label == 'pos':
                cur_file_type = 1
            else:
                cur_file_type = 0

            self.file_path_and_type.append([cur_file_path, cur_file_type])


if __name__ == "__main__":
    dataset = DatasetLoader("/root/official_dataset")
