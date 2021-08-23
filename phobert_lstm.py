import torch
from transformers import AutoModel
import torch.nn as nn
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.phobert(text)[0]

        packed_output, (hidden, cell) = self.rnn(embedded)

        if self.rnn.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        output = self.out(hidden)

        return output


def phobert_lstm(phobert_path='vinai/phobert-base',
                 state_dict_path: str = '',
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 n_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.5,
                 device: torch.device = torch.device('cpu')) -> PhoBERTLSTMSentiment:
    mot cai gi do magic
    model = PhoBERTLSTMSentiment(phobert,
                                 hidden_dim,
                                 num_classes,
                                 n_layers,
                                 bidirectional,
                                 dropout)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)

    return model
