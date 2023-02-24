import numpy as np
import torch.nn as nn
from typing import Iterable

# Replay memory

class ReplayMemory():
    def __init__(self, buffer_limit, obs_dims, obs_dtype, action_dtype):
        self.buffer_limit = buffer_limit
        self.obs_dims = obs_dims
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype
        self.observation = np.empty((buffer_limit, obs_dims), dtype=obs_dtype) 
        self.next_observation = np.empty((buffer_limit, obs_dims), dtype=obs_dtype) 
        self.action = np.empty((buffer_limit, 1), dtype=action_dtype)
        self.reward = np.empty((buffer_limit,), dtype=np.float32) 
        self.terminal = np.empty((buffer_limit,), dtype=bool)
        self.time_step = np.empty((buffer_limit,), dtype=np.uint8)
        self.idx = 0
        self.full = False

    def push(self, transition):
        state, action, reward, next_state, done, time_step = transition
        self.observation[self.idx] = state
        self.next_observation[self.idx] = next_state
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.time_step[self.idx] = time_step
        self.idx = (self.idx + 1) % self.buffer_limit
        self.full = self.full or self.idx == 0
    
    def push_batch(self, transitions, N):
        states, actions, rewards, next_states, dones, time_steps = transitions
        idxs = np.arange(self.idx, self.idx + N) % self.buffer_limit
        self.observation[idxs] = states
        self.next_observation[idxs] = next_states
        self.action[idxs] = actions 
        self.reward[idxs] = rewards
        self.terminal[idxs] = dones
        self.time_step[idxs] = time_steps
        self.full = self.full or (self.idx + N >= self.buffer_limit)    
        self.idx = (idxs[-1] + 1) % self.buffer_limit

    def push_fresh_buffer(self, fresh_buffer):
        N = fresh_buffer.buffer_limit if fresh_buffer.full else fresh_buffer.step_idx
        idxs = np.arange(self.idx, self.idx + N) % self.buffer_limit

        self.observation[idxs] = fresh_buffer.observation[:N]
        self.next_observation[idxs] = fresh_buffer.next_observation[:N]
        self.action[idxs] = fresh_buffer.action[:N]
        self.reward[idxs] = fresh_buffer.reward[:N]
        self.terminal[idxs] = fresh_buffer.terminal[:N]
        self.time_step[idxs] = fresh_buffer.time_step[:N]
        self.full = self.full or (self.idx + N >= self.buffer_limit)
        self.idx = (idxs[-1] + 1) % self.buffer_limit
        
    def sample(self, n):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=n)        
        return self.observation[idxes], self.action[idxes], self.reward[idxes], self.next_observation[idxes], self.terminal[idxes], self.time_step[idxes]

    def __len__(self):
        return self.buffer_limit if self.full else self.idx+1
    
class FreshReplayMemory():
    def __init__(self, num_episodes, max_episode_len, obs_dims, obs_dtype, action_dtype):
        self.obs_dtype = obs_dtype
        self.obs_dims = obs_dims
        self.action_dtype = action_dtype
        self.buffer_limit = num_episodes * max_episode_len
        self.reset()
    
    def reset(self):
        self.observation = np.empty((self.buffer_limit, self.obs_dims), dtype=self.obs_dtype) 
        self.next_observation = np.empty((self.buffer_limit, self.obs_dims), dtype=self.obs_dtype) 
        self.action = np.empty((self.buffer_limit, 1), dtype=self.action_dtype)
        self.reward = np.empty((self.buffer_limit,), dtype=np.float32) 
        self.terminal = np.empty((self.buffer_limit,), dtype=bool)
        self.time_step = np.empty((self.buffer_limit,), dtype=np.uint8)
        self.reward_indices = []
        self.full = False
        self.step_idx = 0
        
    def push(self, transition):
        state, action, reward, next_state, done, time_step = transition
        self.observation[self.step_idx] = state
        self.next_observation[self.step_idx] = next_state
        self.action[self.step_idx] = action 
        self.reward[self.step_idx] = reward
        self.terminal[self.step_idx] = done
        self.time_step[self.step_idx] = time_step

        if done:
            self.reward_indices.append(self.step_idx)
        
        self.step_idx = (self.step_idx + 1) % self.buffer_limit
        self.full = self.full or self.step_idx == 0

    def remove_last_episode(self, episode_len):
        last_episode_start_id = self.step_idx - episode_len
        if self.full : assert self.step_idx == 0
        if last_episode_start_id < 0 : assert self.full
        if last_episode_start_id < 0 : assert self.step_idx == 0
        self.reward_indices.pop()
        self.step_idx = last_episode_start_id % self.buffer_limit
        if self.full : self.full = not last_episode_start_id < 0
        
    def update_final_rewards(self, reward_batch):
        self.reward[self.reward_indices] = reward_batch
 
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