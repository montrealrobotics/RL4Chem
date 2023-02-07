import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import wandb

import utils

class Actor(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, dist='categorical'):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims) 
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, output_dims)
        self.dist = dist

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        if self.dist == 'categorical':
            dist = td.Categorical(logits=logits)
        elif self.dist == 'one_hot_categorical':
            dist = td.OneHotCategoricalStraightThrough(logits=logits)
        else:
            raise NotImplementedError
        return dist 
    
class Critic(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Critic, self).__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(), nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(), nn.Linear(hidden_dims, output_dims))

        self.Q2 = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(), nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(), nn.Linear(hidden_dims, output_dims))

    def forward(self, x):
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2       

class SacAgent:
    def __init__(self, device, obs_dims, num_actions, obs_dtype, action_dtype, env_buffer_size,
                gamma, tau, policy_update_interval, target_update_interval, lr, batch_size, 
                entropy_coefficient, hidden_dims, wandb_log, log_interval):
        '''To-do:
        Try straight through gradients for actor
        Add more implementation details like target actor, noisy td updates, increase critic learning rate
        Replace complicated entropy calculation in update_critic method by direct method from torch distribution
        Thanks author of https://github.com/toshikwa/sac-discrete.pytorch/blob/master/sacd/agent/sacd.py
        '''
        self.device = device

        #learning
        self.gamma = gamma
        self.tau = tau
        self.policy_update_interval = policy_update_interval
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size

        #exploration
        self.entropy_coefficient = entropy_coefficient
        # self.target_entropy = -np.log(1.0 / num_actions) * 0.98

        #logging        
        self.wandb_log = wandb_log
        self.log_interval = log_interval

        self.env_buffer = utils.ReplayMemory(env_buffer_size, obs_dims, obs_dtype, action_dtype)
        self._init_networks(obs_dims, num_actions, hidden_dims)
        self._init_optims(lr)
    
    def get_action(self, obs, step, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device) 
            action_dist = self.actor(obs)    
            action = action_dist.sample()            
            if eval:
                action = action_dist.mode
                
        return action.cpu().numpy()
    
    def update(self, step):
        metrics = dict()
        if step % self.log_interval == 0 and self.wandb_log:
            log = True 
        else:
            log = False 

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.env_buffer.sample(self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)  
        discount_batch = self.gamma*(1-done_batch)

        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, discount_batch, log, metrics)
        actor_log = False
        if step % self.policy_update_interval == 0:
            for _ in range(self.policy_update_interval):
                actor_log = not actor_log if log else actor_log
                self.update_actor(state_batch, actor_log, metrics)
        
        if step%self.target_update_interval==0:
            utils.soft_update(self.critic_target, self.critic, self.tau)

        if log:
            wandb.log(metrics, step=step)  

    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, discount_batch, log, metrics):
        with torch.no_grad():    
            next_action_dist = self.actor(next_state_batch)
            next_action_probs = next_action_dist.probs
            # next_action_entropy = next_action_dist.entropy()

            target_Q1, target_Q2 = self.critic_target(next_state_batch)
            target_V = (next_action_probs * ( torch.min(target_Q1, target_Q2) ) ).sum(dim=1) # + self.alpha * next_action_entropy
           
            target_Q = reward_batch + discount_batch * target_V
            
        Q1, Q2 = self.critic(state_batch)
        Q1 = Q1.gather(1, action_batch.long()).flatten()
        Q2 = Q2.gather(1, action_batch.long()).flatten()

        critic_loss = (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))/2   

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if log:
            metrics['mean_q_target'] = torch.mean(target_Q).item()
            metrics['variance_q_target'] = torch.var(target_Q).item()
            metrics['min_q_target'] = torch.min(target_Q).item()
            metrics['max_q_target'] = torch.max(target_Q).item()
            metrics['critic_loss'] = critic_loss.item()
        
    def update_actor(self, state_batch, log, metrics):

        with torch.no_grad():
            Q1, Q2 = self.critic(state_batch)
            Q = torch.min(Q1, Q2)

        action_dist = self.actor(state_batch)
        action_probs = action_dist.probs
        action_entropy = action_dist.entropy()

        # actor_loss = - (torch.sum(action_probs * Q, dim=1) + self.alpha * action_entropy).mean()
        actor_loss = - (torch.sum(action_probs * Q, dim=1) + self.entropy_coefficient * action_entropy).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # alpha_loss = -torch.mean(self.log_alpha.exp() * (self.target_entropy - action_entropy.detach()))
        # self.alpha_optim.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optim.step()
        # self.alpha = self.log_alpha.exp().item()

        if log:
            metrics['actor_loss'] = actor_loss.item()
            # metrics['alpha_loss'] = alpha_loss.item()
            # metrics['alpha'] = self.alpha
            metrics['actor_entropy'] = action_entropy.detach().mean().item()

    def _init_networks(self, obs_dims, num_actions, hidden_dims):
        self.actor = Actor(obs_dims, hidden_dims, num_actions).to(self.device)

        self.critic = Critic(obs_dims, hidden_dims, num_actions).to(self.device)
        self.critic_target = Critic(obs_dims, hidden_dims, num_actions).to(self.device)
        utils.hard_update(self.critic_target, self.critic)

        # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        # self.alpha = self.log_alpha.exp().item()

    def _init_optims(self, lr):
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr['actor'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr['critic'])
        # self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr['alpha'])

    def get_save_dict(self):
        return {
            "critic": self.critic.state_dict(),
            "critic_target":self.critic_target.state_dict(),
            "actor": self.actor.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.critic.load_state_dict(saved_dict["critic"])
        self.critic_target.load_state_dict(saved_dict['critic_target'])
        self.actor.load_state_dict(saved_dict['actor'])