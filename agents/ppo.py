import gym
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
        
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        num_states = np.array(envs.single_observation_space.shape).prod()
        num_actions = envs.single_action_space.n

        self.envs = envs
        self.cfg = cfg
        self.device = device

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

        self.actorcritic_list = [self.actor, self.critic]
        self.optimizer = optim.Adam(get_parameters(self.actorcritic_list), lr=cfg.lr, eps=1e-5)

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

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

    def train(self, obs, actions, logprobs, rewards, values, next_obs, dones, next_done):     
        cfg = self.cfg 

        advantages, returns = self.get_return(rewards, values, next_obs, dones, next_done)

        # flatten the batch
        b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

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