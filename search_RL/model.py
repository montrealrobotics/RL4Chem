import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as td

class mlp_actor():
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(mlp_actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims) 
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        dist = td.Categorical(logits=logits)
        return dist
    
    def sample_sequence(self, start_tokens, max_len):
        sequences = []
        x = start_tokens
        for step in range(max_len):
            dist = self(x)