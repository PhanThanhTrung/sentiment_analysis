import torch.nn as nn
import torch
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, hidden_dim: int, num_classes: int, n_layers: int, 
                 bidirectional: bool, dropout: float):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first= True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        
    def forward(self,x,text_lengths):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x, text_lengths.to('cpu'), batch_first= True,enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output,batch_first= True)
        
        hidden = hidden[-2:, :, :]
        hidden = torch.cat((hidden[0],hidden[1]), dim= 1)
        hidden = self.dropout(hidden)
        fc = self.fc(hidden)
        return fc