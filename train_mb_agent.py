import os
import time
import wandb
import hydra
import torch
import torch.nn.functional as F
import random
import numpy as np
from omegaconf import DictConfig
from optimizer import BaseOptimizer
from pathlib import Path
path_here = os.path.dirname(os.path.realpath(__file__))

from models.mb import RnnPolicy, RnnReward
from data import smiles_vocabulary, selfies_vocabulary

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_params(modules):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()
    
class ReplayBuffer():
    def __init__(self, vocab, max_size, max_len):

        self.max_size = max_size
        self.max_len = max_len
        self.vocab = vocab
        
        self.obs = np.ones((max_len, max_size), dtype=np.uint8) * self.vocab.pad
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.ep_len = np.zeros((max_size,), dtype=np.int64) 

        self.mol_idx = 0
        
        self.num_episodes = 0        
        self.full = False

    def add_experience(self, obs, score, ep_len):
        #getting indices to store new data
        L, N = obs.shape
        mol_idxs = np.arange(self.mol_idx, self.mol_idx + N) % self.max_size

        #clearing old placeholders
        self.obs[:, mol_idxs] = self.vocab.pad

        #create masks
        idx = np.broadcast_to(np.expand_dims(np.arange(L), axis=1), (L, N))
        idx = idx < ep_len + 1

        self.obs[:L, mol_idxs] = idx * obs + ~idx * self.vocab.pad

        self.rewards[mol_idxs] = score
        self.ep_len[mol_idxs] = ep_len

        self.full = self.full or (self.mol_idx + N >= self.max_size)    
        self.mol_idx = (mol_idxs[-1] + 1) % self.max_size

    def sample(self, n, device):
        mol_idxs = np.random.choice(len(self), size=n)

        ep_len = self.ep_len[mol_idxs]
        maxlen_batch = max(ep_len)

        return torch.tensor(self.obs[:maxlen_batch+1, mol_idxs],  dtype=torch.long, device=device), \
            torch.tensor(self.rewards[mol_idxs], dtype=torch.float32, device=device), \
            torch.tensor(ep_len)

    def __len__(self):
        return int(self.max_size if self.full else self.mol_idx)

class mb_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent_name = cfg.agent_name

    def _init(self, cfg):
        if cfg.dataset == 'zinc250k':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc250k/zinc_' + cfg.rep + '_vocab.txt'

        #get data
        if cfg.rep == 'smiles':
            self.vocab = smiles_vocabulary(vocab_path=os.path.join(path_here, vocab_path))
        elif cfg.rep == 'selfies':
            self.vocab = selfies_vocabulary(vocab_path=os.path.join(path_here, vocab_path))
        else:
            raise NotImplementedError

        if cfg.model_name == 'transformer':
            raise NotImplementedError
        elif cfg.model_name == 'char_rnn':
            #get prior
            prior_saved_dict = torch.load(os.path.join(path_here, saved_path))
 
            # get agent
            self.actor = RnnPolicy(self.vocab, cfg.embedding_size, cfg.hidden_size, cfg.num_layers).to(self.device)
            self.actor.load_save_dict(prior_saved_dict)

            self.reward_model = RnnReward(self.vocab, cfg.embedding_size, cfg.hidden_size, cfg.num_layers).to(self.device)
            self.reward_model.load_save_dict(prior_saved_dict)
    
        else:
            raise NotImplementedError
    
        # get optimizers
        self.actor_optimizer = torch.optim.Adam(get_params([self.actor]), lr=cfg['learning_rate'])
        self.reward_optimizer = torch.optim.Adam(get_params([self.reward_model]), lr=cfg['learning_rate'])

        # get replay memory
        self.replay_memory = ReplayBuffer(self.vocab, cfg.max_strings, cfg.max_len+1)
    
    def update_reward(self, cfg, metrics, v_obs, v_rewards, v_episode_lens, log):
        for i in range(5):
            obs, rewards, episode_lens = self.replay_memory.sample(cfg.reward_batch_size, self.device)

            #reward loss
            reward_preds = self.reward_model(obs, episode_lens)
            reward_loss = F.mse_loss(reward_preds, rewards)

            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            reward_grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 0.5)
            self.reward_optimizer.step() 

        if log:
            with torch.no_grad():
                v_reward_preds  = self.reward_model(v_obs, v_episode_lens)
                v_reward_loss = F.l1_loss(v_reward_preds, v_rewards)
            metrics['v_reward_l1_loss'] = v_reward_loss
            metrics['reward_grad_norm'] = reward_grad_norm.item()
            metrics['reward_loss'] = reward_loss
        return metrics

    def update_policy(self, cfg, metrics, log):
        with torch.no_grad():
            # sample experience
            obs, rewards, old_logprobs, nonterms, episode_lens = self.actor.get_data(cfg.batch_size, cfg.max_len, self.device)
            scores = self.reward_model(obs, episode_lens)

        rewards = rewards * scores
        rev_returns = torch.cumsum(rewards, dim=0) 
        advantages = rewards - rev_returns + rev_returns[-1:]
        
        #policy loss
        logprobs = self.actor.get_likelihood(obs, episode_lens, nonterms)
        logratio = logprobs - old_logprobs
        ratio = logratio.exp()
        loss_pg1 = -advantages * ratio
        loss_pg2 = -advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)

        loss_pg = torch.max(loss_pg1, loss_pg2).sum(0, keepdim=True).mean()
        loss_p = - (1 / logprobs.sum(0, keepdim=True)).mean()

        #total loss
        loss = loss_pg + cfg.lp_coef * loss_p

        # Calculate gradients and make an update to the network weights
        self.actor_optimizer.zero_grad()
        loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        if log:
            with torch.no_grad():
                clipfracs = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                policy_kl = masked_mean(-logratio.mean(), nonterms[:-1]).item()
                
            metrics['pg_loss'] = loss_pg.item()       
            metrics['agent_likelihood'] = logprobs.sum(0).mean().item()
            metrics['agent_old_likelihood'] = old_logprobs.sum(0).mean().item()
            metrics['actor_grad_norm'] = actor_grad_norm.item() 
            
            metrics['smiles_len'] = episode_lens.float().mean().item()
    
            metrics['clipfracs'] = clipfracs
            metrics['policy_kl'] = policy_kl            
            
            metrics['loss_p'] = loss_p.item()
        return metrics

    def optimize(self, cfg):
        if cfg.wandb_log:
            self.define_wandb_metrics()

        #set device
        self.device = torch.device(cfg.device)

        self._init(cfg)

        train_steps = 0
        eval_strings = 0
        metrics = dict() 
        while eval_strings < cfg.max_strings:
            with torch.no_grad():
                # sample experience
                obs, episode_lens = self.actor.sample(cfg.batch_size, cfg.max_len, self.device)

            smiles_list = []
            for en_sms in obs.cpu().numpy().T:
                sms = self.vocab.decode_padded(en_sms)
                smiles_list.append(sms)
                
            score = np.array(self.predict(smiles_list))
            scores = torch.tensor(score, dtype=torch.float32, device=self.device).unsqueeze(0)

            if self.finish:
                print('max oracle hit')
                break 

            train_steps += 1
            eval_strings += cfg.batch_size

            log = False
            if cfg.wandb_log and train_steps % cfg.train_log_interval == 0:
                log = True
                metrics = dict()
                metrics['eval_strings'] = eval_strings
                metrics['mean_score'] = np.mean(score)
                metrics['max_score'] = np.max(score)
                metrics['min_score'] = np.min(score)
                metrics['mean_episode_lens'] = np.mean(episode_lens.tolist())
                metrics['max_episode_lens'] = np.max(episode_lens.tolist())
                metrics['min_episode_lens'] = np.min(episode_lens.tolist())
            
            if eval_strings > cfg.warmup_strings:
                # train reward function
                metrics = self.update_reward(cfg, metrics, obs, scores, episode_lens, log)

                # train the policy
                metrics = self.update_policy(cfg, metrics, log)
        
            if log:
                wandb.log(metrics)
            
            self.replay_memory.add_experience(obs.cpu().numpy(), score, episode_lens.numpy())

@hydra.main(config_path='cfgs', config_name='mb', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    if cfg.wandb_log:
        project_name = cfg.task + '_' + cfg.target
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg))
        wandb.run.name = cfg.wandb_run_name
    
    set_seed(cfg.seed)
    cfg.output_dir = hydra_cfg['runtime']['output_dir']

    optimizer = mb_optimizer(cfg)
    optimizer.optimize(cfg)

if __name__ == '__main__':
    main()