import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class CharRNN(nn.Module):
    def __init__(self, vocabulary, device):
        super(CharRNN, self).__init__()

        self.vocabulary = vocabulary
        self.hidden_size = 256
        self.num_layers = 3
        self.dropout = 0.2
        self.device = device
        self.vocab_size = self.input_size = len(vocabulary)

        self.embedding_layer = nn.Embedding(self.vocab_size, 32,
                                            padding_idx=vocabulary.pad).to(self.device)
        self.lstm_layer = nn.LSTM(32, self.hidden_size,
                                  self.num_layers, dropout=0.2,
                                  batch_first=True).to(self.device)
        self.linear_layer = nn.Linear(self.hidden_size, 1).to(self.device)

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)        
        _, (h_last, _) = self.lstm_layer(x, hiddens)
        preds = self.linear_layer(h_last[-1]).squeeze(-1)
        return preds