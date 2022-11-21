import gym
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import layer_init, get_parameters
from torch.distributions.categorical import Categorical

class PpoAgent(object):
    def __init__(self, envs, cfg, device):
        self._init_agent_networks(device, envs)
        self._init_agent_optim(cfg)
        self._init_agent_memory(device, envs, cfg)
        
        self.cfg = cfg
        self.device = device

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def collect_data(self, train_envs, train_step, train_episode, start_time):
        cfg = self.cfg
        device = self.device

        for step in range(0, cfg.num_steps):
            train_step += 1 * cfg.num_envs
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done

            with torch.no_grad():
                action, logprob, _, value = self.get_action_and_value(self.next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            next_obs, reward, done, info = train_envs.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(device).view(-1)
            self.next_obs, self.next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    train_episode += 1
                    print("total episodes: {}, total numsteps: {}, return: {}, cummulative sps: {}".format(train_episode, train_step, item["episode"]["r"], int(train_step / (time.time() - start_time))))
                    if cfg.wandb_log:
                        episode_metrics = dict()
                        episode_metrics['episodic_length'] = item["episode"]["l"]
                        episode_metrics['episodic_return'] = item["episode"]["r"]
                        wandb.log(episode_metrics, step=train_step)
                    break
        
        return train_step, train_episode

    def get_return(self, rewards, values, next_obs, dones, next_done):
        cfg = self.cfg
        device = self.device
        #calculate returns
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            if cfg.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + cfg.gamma * nextnonterminal * next_return
                advantages = returns - values
        return advantages, returns

    def train(self, envs):     
        cfg = self.cfg 

        advantages, returns = self.get_return(self.rewards, self.values, self.next_obs, self.dones, self.next_done)

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(get_parameters(self.actorcritic_list), cfg.max_grad_norm)
                self.optimizer.step()

            if cfg.target_kl is not None:
                if approx_kl > cfg.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        metrics = dict()
        metrics['learning_rate'] = self.optimizer.param_groups[0]["lr"]
        metrics['value_loss'] = v_loss.item()
        metrics['policy_loss'] = pg_loss.item()
        metrics['entropy'] = entropy_loss.item()
        metrics['old_approx_kl'] = old_approx_kl.item()
        metrics['approx_kl'] = approx_kl.item()
        metrics['clipfrac'] = np.mean(clipfracs)
        metrics['explained_variance'] = explained_var
        return metrics
        
    def _init_agent_memory(self, device, envs, cfg):
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        self.obs = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        self.rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        self.dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        self.values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        
        self.next_obs = torch.Tensor(envs.reset()).to(device)
        self.next_done = torch.zeros(cfg.num_envs).to(device)

    def _init_agent_networks(self, device, envs):
        num_states = np.array(envs.single_observation_space.shape).prod()
        num_actions = envs.single_action_space.n
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_states, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(device)
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(num_states, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        ).to(device)

    def _init_agent_optim(self, cfg):
        self.actorcritic_list = [self.actor, self.critic]
        self.optimizer = optim.Adam(get_parameters(self.actorcritic_list), lr=cfg.lr, eps=1e-5)