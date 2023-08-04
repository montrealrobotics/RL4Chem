import os
import sys
import wandb
import hydra
import torch
import random
import numpy as np
import selfies as sf
import torch.nn.functional as F
from omegaconf import DictConfig
from optimizer import BaseOptimizer
path_here = os.path.dirname(os.path.realpath(__file__))

from models.ac import RnnPolicy, RnnValue
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

class ac_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent_name = cfg.agent_name

    def _init(self, cfg):
        if cfg.dataset == 'zinc250k':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc250k/zinc_' + cfg.rep + '_vocab.txt'
            max_dataset_len = 73
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        else:
            raise NotImplementedError
        
        #get data
        if cfg.rep == 'smiles':
            self.vocab = smiles_vocabulary(vocab_path=os.path.join(path_here, vocab_path))
        elif cfg.rep == 'selfies':
            self.vocab = selfies_vocabulary(vocab_path=os.path.join(path_here, vocab_path))
        else:
            raise NotImplementedError
       
        assert cfg.model_name == 'char_rnn'
        #get prior
        prior_saved_dict = torch.load(os.path.join(path_here, saved_path))

        self.actor = RnnPolicy(self.vocab, cfg.embedding_size, cfg.hidden_size, cfg.num_layers).to(self.device)
        self.actor.load_save_dict(prior_saved_dict)

        self.critic = RnnValue(self.vocab, cfg.embedding_size, cfg.hidden_size, cfg.num_layers).to(self.device)
        self.critic.load_save_dict(prior_saved_dict)

        # get optimizers
        self.optimizer = torch.optim.Adam(get_params([self.actor, self.critic]), lr=cfg['learning_rate'])

    def update(self, obs, rewards, old_values, nonterms, episode_lens, cfg, metrics, log):
        # rev_returns = torch.cumsum(rewards, dim=0) 
        # advantages = rewards - rev_returns + rev_returns[-1:]

        max_episode_len = episode_lens.max()
        with torch.no_grad():
            advantages_new = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(max_episode_len)):
                if t == max_episode_len - 1:
                    delta = rewards[t] - old_values[t]
                else:
                    delta = rewards[t] + old_values[t + 1] - old_values[t]
                
                advantages_new[t] = lastgaelam = delta + 0.95 * nonterms[t + 1] * lastgaelam
            returns_new = advantages_new + old_values

        logprobs = self.actor.get_likelihood(obs, episode_lens, nonterms)
        loss_pg = -advantages_new * logprobs
        loss_pg = loss_pg.sum(0, keepdim=True).mean()
        loss_p = - (1 / logprobs.sum(0, keepdim=True)).mean()

        #value loss
        values = self.critic.get_value(obs, episode_lens, nonterms)
        v_loss_unclipped = F.mse_loss(values, returns_new)
        v_clipped = old_values + torch.clamp(values - old_values, -cfg.clip_coef, cfg.clip_coef)
        v_loss_clipped  = F.mse_loss(v_clipped, returns_new)
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
        
        #total loss
        loss = loss_pg + cfg.lp_coef * loss_p + v_loss

        # Calculate gradients and make an update to the network weights
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) + torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

        if log:
            metrics['pg_loss'] = loss_pg.item()       
            metrics['agent_likelihood'] = logprobs.sum(0).mean().item()
            metrics['grad_norm'] = grad_norm.item() 
            metrics['smiles_len'] = episode_lens.float().mean().item()
            metrics['loss_p'] = loss_p.item()
            metrics['v_loss'] = v_loss
            
            print('logging!')
            wandb.log(metrics)

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
                obs, rewards, values, nonterms, episode_lens = self.actor.get_data(self.critic, cfg.batch_size, cfg.max_len, self.device)

            if cfg.rep == 'selfies':
                smiles_list = []
            
                if cfg.penalty == 'diff':
                    true_selfies_len_list = []
                    for i, en_sms in enumerate(obs.cpu().numpy().T):
                        sms = self.vocab.decode_padded(en_sms)
                        try:
                            true_selfies_len_list.append(sf.len_selfies(sf.encoder(sms)))
                        except:
                            true_selfies_len_list.append(episode_lens[i])
                        smiles_list.append(sms)
                    # difference penalty
                    score = np.array(self.predict(smiles_list)) - np.abs(episode_lens.numpy() - true_selfies_len_list) / cfg.max_len
                    scores = torch.tensor(score, dtype=torch.float32, device=self.device).unsqueeze(0)
                elif cfg.penalty == 'len':
                    smiles_list = []
                    for en_sms in obs.cpu().numpy().T:
                        sms = self.vocab.decode_padded(en_sms)
                        smiles_list.append(sms)

                    # length penalty
                    score = np.array(self.predict(smiles_list)) - episode_lens.numpy() / cfg.max_len
                    scores = torch.tensor(score, dtype=torch.float32, device=self.device).unsqueeze(0) 
                else:
                    smiles_list = []
                    for en_sms in obs.cpu().numpy().T:
                        sms = self.vocab.decode_padded(en_sms)
                        smiles_list.append(sms)

                    # no penalty
                    score = np.array(self.predict(smiles_list))
                    scores = torch.tensor(score, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                smiles_list = []
                for en_sms in obs.cpu().numpy().T:
                    sms = self.vocab.decode_padded(en_sms)
                    smiles_list.append(sms)

                score = np.array(self.predict(smiles_list))
                scores = torch.tensor(score, dtype=torch.float32, device=self.device).unsqueeze(0)

            if self.finish:
                print('max oracle hit')
                wandb.finish()
                sys.exit(0)

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
                wandb.log(metrics)

            rewards = rewards * scores
            self.update(obs, rewards, values, nonterms, episode_lens, cfg, metrics, log)
        
        print('max training string hit')
        wandb.finish()
        sys.exit(0)


@hydra.main(config_path='cfgs', config_name='ac_rnn', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    if cfg.wandb_log:
        project_name = cfg.task + '_' + cfg.target
        if cfg.wandb_dir is not None:
            cfg.wandb_dir = path_here 
        else:
            cfg.wandb_dir = hydra_cfg['runtime']['output_dir']
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=cfg.wandb_dir)
        wandb.run.name = cfg.wandb_run_name
    
    set_seed(cfg.seed)
    cfg.output_dir = hydra_cfg['runtime']['output_dir']

    optimizer = ac_optimizer(cfg)
    optimizer.optimize(cfg)
    sys.exit(0)

if __name__ == '__main__':
    main()
    exit()        
