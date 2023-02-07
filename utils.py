import numpy as np
import torch.nn as nn
from typing import Iterable

# Replay memory

class ReplayMemory():
    def __init__(self, buffer_limit, obs_dims, obs_dtype, action_dtype):
        print('buffer limit is = ', buffer_limit)
        self.buffer_limit = buffer_limit
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype
        self.observation = np.empty((buffer_limit, obs_dims), dtype=obs_dtype) 
        self.next_observation = np.empty((buffer_limit, obs_dims), dtype=obs_dtype) 
        self.action = np.empty((buffer_limit, 1), dtype=action_dtype)
        self.reward = np.empty((buffer_limit,), dtype=np.float32) 
        self.terminal = np.empty((buffer_limit,), dtype=bool)
        self.idx = 0
        self.full = False

    def push(self, transition):
        state, action, reward, next_state, done = transition
        self.observation[self.idx] = state
        self.next_observation[self.idx] = next_state
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.buffer_limit
        self.full = self.full or self.idx == 0
    
    def push_batch(self, transitions, N):
        states, actions, rewards, next_states, dones = transitions
        idxs = np.arange(self.idx, self.idx + N) % self.buffer_limit
        self.observation[idxs] = states
        self.next_observation[idxs] = next_states
        self.action[idxs] = actions 
        self.reward[idxs] = rewards
        self.terminal[idxs] = dones
        self.idx = (idxs[-1] + 1) % self.buffer_limit
        self.full = self.full or 0 in idxs

    def sample(self, n):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=n)
        return self.observation[idxes], self.action[idxes], self.reward[idxes], self.next_observation[idxes], self.terminal[idxes]

    def __len__(self):
        return self.buffer_limit if self.full else self.idx+1

# NN weight utils

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# NN module utils

def get_parameters(modules: Iterable[nn.Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

#schedule utils

def linear_schedule(start_sigma: float, end_sigma: float, duration: int, t: int):
    return end_sigma + (1 - min(t / duration, 1)) * (start_sigma - end_sigma)