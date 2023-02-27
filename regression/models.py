import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CharMLP(nn.Module):
    def __init__(self, vocab_size, pad_idx, embedding_size, input_size, hidden_size, device):
        super(CharMLP, self).__init__()
        self.device = device
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx).to(self.device)
        self.fc_layer = nn.Sequential(
            nn.Linear(input_size * embedding_size, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, 1),
        ).to(self.device)

    def forward(self, x):
        x = self.embedding_layer(x).view(-1, self.input_size * self.embedding_size)
        preds = self.fc_layer(x).squeeze(-1)
        return preds
    
class CharConv(nn.Module):
    def __init__(self, vocab_size, pad_idx, embedding_size, input_size, hidden_size, device):
        super(CharConv, self).__init__()
        self.device = device
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx).to(self.device).to(self.device)
        self.convnet = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.embedding_size, kernel_size=10).to(self.device)
        self.temp = (self.input_size - 10 + 1) * self.embedding_size
        self.linear_layer_1 = nn.Linear(self.temp, hidden_size).to(self.device)
        self.linear_layer_2 = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x):
        x = self.embedding_layer(x).view(-1, self.input_size * self.embedding_size).view(-1, self.embedding_size, self.input_size)        
        x = F.relu(self.convnet(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear_layer_1(x))
        preds = self.linear_layer_2(x).squeeze(-1)
        return preds
        
class CharRNN(nn.Module):
    def __init__(self, vocab_size, pad_idx, embedding_size, num_layers, hidden_size, device):
        super(CharRNN, self).__init__()
        self.device = device
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx).to(self.device)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.linear_layer = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)        
        _, (h_last, _) = self.lstm_layer(x, hiddens)
        preds = self.linear_layer(h_last[-1]).squeeze(-1)
        return preds