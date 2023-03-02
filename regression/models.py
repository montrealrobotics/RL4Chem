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
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx).to(self.device)
        self.conv1d1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=9, kernel_size=9).to(self.device)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9).to(self.device)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11).to(self.device)

        self.temp = (self.input_size - 26) * 10
        self.linear_layer_1 = nn.Linear(self.temp, 256).to(self.device)
        self.linear_layer_2 = nn.Linear(256, 1).to(self.device)

    def forward(self, x):
        x = torch.transpose(self.embedding_layer(x), 1, 2)
        x = F.relu(self.conv1d1(x))
        x = F.relu(self.conv1d2(x))
        x = F.relu(self.conv1d3(x))
        x = x.view(x.shape[0], -1)
        x = F.selu(self.linear_layer_1(x))
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