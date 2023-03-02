import time
import hydra
import random
import numpy as np

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as td
import torch.nn.utils.rnn as rnn_utils

from omegaconf import DictConfig
from collections import defaultdict

import utils

class Actor(nn.Module):
    def __init__(self, vocab_size, pad_idx, embedding_size, num_layers, hidden_size, output_size, device, dist='categorical'):
        super(Actor, self).__init__()
        self.device = device
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx).to(self.device)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.linear_layer = nn.Linear(hidden_size, output_size).to(self.device)
    
    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, hiddens = self.lstm_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        logits = self.linear_layer(x)
        print(logits.shape)
        exit()
        return logits
      
class Reward(nn.Module):
    def __init__(self, vocab_size, pad_idx, embedding_size, num_layers, hidden_size, dropout, device):
        super(Reward, self).__init__()
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

def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)

@hydra.main(config_path='cfgs', config_name='mbac', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    if cfg.wandb_log:
        import wandb
        project_name = 'rl4chem_' + cfg.target
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=hydra_cfg['runtime']['output_dir'])
        wandb.run.name = cfg.wandb_run_name

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == 'cuda' else "cpu")

    #set seed 
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    #get vocab
    from env import selfies_vocabulary
    vocab = selfies_vocabulary()

    #get replay buffer
    rb = utils.ReplayBuffer(cfg.env_buffer_size)

    #get model
    cfg.vocab_size = len(vocab)
    cfg.pad_idx = vocab.pad
    actor_model = hydra.utils.instantiate(cfg.actor)
    reward_model = hydra.utils.instantiate(cfg.reward)

    #set optimizer
    actor_optimizer = torch.optim.Adam(get_params(actor_model), lr=cfg.lr['actor'])
    reward_optimizer = optim.Adam(get_params(reward_model), lr=cfg.lr['reward'])

    max_length = 100
    train_step = 0
    while train_step < cfg.num_train_steps:
        with torch.no_grad():
            starts = torch.full((cfg.parallel_molecules, 1), fill_value=vocab.bos, dtype=torch.long, device=device)
            
            new_smiles_list = [
                torch.tensor(vocab.pad, dtype=torch.long,
                             device=device).repeat(max_length + 2)
                for _ in range(cfg.parallel_molecules)]
            
            print(new_smiles_list)
            exit()
           

if __name__ == "__main__":
    main()