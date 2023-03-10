import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import random
import wandb

import utils

class QNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dims, hidden_dims), 
            nn.ReLU(), nn.Linear(hidden_dims, hidden_dims), nn.LayerNorm(hidden_dims),
            nn.ReLU(), nn.Linear(hidden_dims, output_dims))

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, device, obs_dims, num_actions,
                gamma, tau, update_interval, target_update_interval, lr, batch_size, 
                hidden_dims, wandb_log, log_interval):
        
        self.device = device

        #learning
        self.gamma = gamma
        self.tau = tau
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size

        #logging        
        self.wandb_log = wandb_log
        self.log_interval = log_interval

        self._init_networks(obs_dims, num_actions, hidden_dims)
        self._init_optims(lr)
    
    def get_action(self, obs, eval=False):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            q_values = self.q(obs)
            action = torch.argmax(q_values)                
        return action.cpu().numpy()

    def _init_networks(self, obs_dims, num_actions, hidden_dims):
        self.q = QNetwork(obs_dims, hidden_dims, num_actions).to(self.device)
        self.q_target = QNetwork(obs_dims, hidden_dims, num_actions).to(self.device)
        utils.hard_update(self.q_target, self.q)

    def _init_optims(self, lr):
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=lr["q"])

    def get_save_dict(self):
        return {
            "q": self.q.state_dict(),
            "q_target":self.q_target.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.q.load_state_dict(saved_dict["q"])
        self.q_target.load_state_dict(saved_dict["q_target"])

    def update(self, buffer, step):
        metrics = dict()
        if step % self.log_interval == 0 and self.wandb_log:
            log = True 
        else:
            log = False

        if step % self.update_interval == 0:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, time_batch = buffer.sample(self.batch_size)
            state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
            next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
            action_batch = torch.tensor(action_batch,  dtype=torch.long, device=self.device)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
            done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)  
            discount_batch = self.gamma*(1-done_batch)

            with torch.no_grad():
                target_max, _ = self.q_target(next_state_batch).max(dim=1)
                td_target = reward_batch + self.gamma * target_max * discount_batch
               
            old_val = self.q(state_batch).gather(1, action_batch).squeeze()

            loss = F.mse_loss(td_target, old_val)
            self.q_opt.zero_grad()
            loss.backward()
            self.q_opt.step()

            if log:
                metrics['mean_q_target'] = torch.mean(td_target).item()
                metrics['max_reward'] = torch.max(reward_batch).item()
                metrics['min_reward'] = torch.min(reward_batch).item()
                metrics['variance_q_target'] = torch.var(td_target).item()
                metrics['min_q_target'] = torch.min(td_target).item()
                metrics['max_q_target'] = torch.max(td_target).item()
                metrics['critic_loss'] = loss.item()
            
        if step % self.target_update_interval == 0:
            utils.soft_update(self.q_target, self.q, self.tau)
        
        if log:
            wandb.log(metrics, step=step)  