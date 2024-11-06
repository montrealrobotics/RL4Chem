import os
import sys
import wandb
import hydra
import torch
import random
import numpy as np
import torch.optim as optim
from omegaconf import DictConfig
from optimizer import BaseOptimizer
path_here = os.path.dirname(os.path.realpath(__file__))

from models.reinforce import TransPolicy
from data import smiles_vocabulary, selfies_vocabulary

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_params(model):
    return (p for p in model.parameters() if p.requires_grad)

def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

class reinforce_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent_name = cfg.agent_name

    def _init(self, cfg):
        if cfg.dataset == 'molgen_oled_1':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + 'chembl0.25_zinc0.25_moses0.25_oled0.25.pt'
            vocab_path = 'data/molgen_oled_1/molgen_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 366 # 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        elif cfg.dataset == 'molgen_oled_2':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + 'chembl0.2_zinc0.2_moses0.2_oled0.4.pt'
            vocab_path = 'data/molgen_oled_2/molgen_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 366 # 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        elif cfg.dataset == 'molgen':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + 'chembl0.1_zinc0.3_moses0.6.pt'
            vocab_path = 'data/molgen/molgen_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
            
        elif cfg.dataset == 'chembl':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/chembl/chembl_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 112
            elif cfg.rep=='selfies':
                max_dataset_len = 106
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        
        elif cfg.dataset == 'zinc250k':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc250k/zinc_' + cfg.rep + '_vocab.txt'
            max_dataset_len = 73
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        
        elif cfg.dataset == 'zinc1m':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc1m/zinc_' + cfg.rep + '_vocab_1M.txt'
            max_dataset_len = 74
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        
        elif cfg.dataset == 'zinc10m':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc10m/zinc_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 85
            elif cfg.rep=='selfies':
                max_dataset_len = 88
            if cfg.max_len > max_dataset_len:
                cfg.max_len = max_dataset_len
                print('*** Changing the maximum length of sampled molecules because it was set to be greater than the maximum length seen during training ***')
        
        elif cfg.dataset == 'zinc100m':
            saved_path = 'saved/' + cfg.dataset + '/' + cfg.model_name + '_' + cfg.rep + '/' + cfg.saved_name
            vocab_path = 'data/zinc100m/zinc_' + cfg.rep + '_vocab.txt'
            if cfg.rep=='smiles': 
                max_dataset_len = 85
            elif cfg.rep=='selfies':
                max_dataset_len = 88
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
    
        print('Vocab assigned')

        self.target_entropy = - 0.98 * torch.log(1 / torch.tensor(len(self.vocab)))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=3e-4, eps=1e-4)
        self.highest_scored_mols = []

        assert cfg.model_name == 'char_trans'
        #get prior
        print('Saved path: ', saved_path)
        print('Path here: ', path_here)
        prior_saved_dict = torch.load(os.path.join(path_here, saved_path))
        print('Prior loaded')

        # get agent
        self.agent = TransPolicy(self.vocab, max_dataset_len, cfg.n_heads, cfg.n_embed, cfg.n_layers, dropout=cfg.dropout)
        
        print('Agent class initialised')

        self.agent.to(self.device)

        print('Agent class transferred to cuda memory')

        self.agent.load_save_dict(prior_saved_dict)

        print('Prior weights initialised')

        # get optimizers
        self.optimizer = torch.optim.Adam(get_params(self.agent), lr=cfg['learning_rate'])

        print('Initialisation of optimizer is done!')
    
    def update(self, obs, rewards, nonterms, episode_lens, cfg, metrics, log):
        rev_returns = torch.cumsum(rewards, dim=0) 
        advantages = rewards - rev_returns + rev_returns[-1:]

        logprobs, log_of_probs, action_probs = self.agent.get_likelihood(obs, nonterms)

        # print(logprobs)
        # print(act_probs)
        # print(logprobs.shape)
        # print(act_probs.shape)
        # exit()

        loss_pg = -advantages * logprobs
        loss_pg = loss_pg.sum(0, keepdim=True).mean()
        
        
        #loss_p = - (1 / logprobs.sum(0, keepdim=True)).mean()
        loss = loss_pg #+ cfg.lp_coef * loss_p 
        loss = loss_pg + self.alpha * logprobs.sum(0, keepdim=True).mean()

        # Calculate gradients and make an update to the network weights
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()

        alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_of_probs + self.target_entropy).detach())).mean()
        
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        if log:
            metrics['pg_loss'] = loss_pg.item()       
            metrics['agent_likelihood'] = logprobs.sum(0).mean().item()
            metrics['grad_norm'] = grad_norm.item() 
            metrics['smiles_len'] = episode_lens.float().mean().item()
            # metrics['loss_p'] = loss_p.item()
            metrics['alpha'] = self.alpha
            metrics['alpha_loss'] = alpha_loss.detach().item()
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
        print('Start training ... ')
        while eval_strings < cfg.max_strings:

            with torch.no_grad():
                # sample experience
                obs, rewards, nonterms, episode_lens = self.agent.get_data(cfg.batch_size, cfg.max_len, self.device)

            smiles_list = []
            for en_sms in obs.cpu().numpy().T:
                sms = self.vocab.decode_padded(en_sms)
                smiles_list.append(sms)
            #smiles_list = ['COc1ccc(N2CCC3CC2c2cc(-c4ccc(Cl)c(C#N)c4)ccc23)cc1', 'N#CCn1cnc2c(-c3ccc(F)cc3)csc2c1=O', 'OC1CN(Cc2ccncc2)CC(O)C1N1CCOCC1', 'C=C(C)CN(CC)S(=O)(=O)c1c(C)cc(C)cc1C', 'c1ccc2[nH]c(CN3CCN(Cc4ccon4)CC3)nc2c1', 'CC1(C)CCCC(C)(C)N1c1cc(-c2nccc3nccnc23)c(-c2cncc3nccnc23)cc1-n1c2ccccc2c2ccccc21', 'COC(=O)C(CC(C)C)NC(=O)c1ccc(OC)nc1', 'COc1nc2ccc(Br)cc2cc1C(c1ccc(C)cc1)n1ccnc1', 'C=CCN(C(=O)c1ccc2nnnn2c1)c1nc(-c2ccco2)cs1', 'CCn1nc(C)c(CNc2ccc(F)cc2C(=O)N2CCOCC2)c1C']
            score, logp_scores = np.array(self.predict(smiles_list))
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
                metrics['mean_logp_score'] = np.mean(logp_scores)
                metrics['max_logp_score'] = np.max(logp_scores)
                metrics['min_logp_score'] = np.min(logp_scores)
                metrics['mean_episode_lens'] = np.mean(episode_lens.tolist())
                metrics['max_episode_lens'] = np.max(episode_lens.tolist())
                metrics['min_episode_lens'] = np.min(episode_lens.tolist())
                wandb.log(metrics)
                self.update_highest_scored_molecules(logp_scores, smiles_list)

            rewards = rewards * scores
            self.update(obs, rewards, nonterms, episode_lens, cfg, metrics, log)

        print('max training string hit')
        print('Highest scored mols:', self.highest_scored_mols)
        wandb.finish()
        sys.exit(0)

    def update_highest_scored_molecules(self, scores, molecules):
        for idx, new_score in enumerate(scores):
            mol = molecules[idx]
            existing_mol = next((m for m in self.highest_scored_mols if m['mol'] == mol), None)

            if existing_mol:
                if new_score > existing_mol['score']:
                    existing_mol['score'] = new_score
            else:
                if len(self.highest_scored_mols) < 5:
                    self.highest_scored_mols.append({'mol': mol, 'score': new_score})
                else:
                    lowest_score = min(self.highest_scored_mols, key=lambda x: x['score'])

                    if new_score > lowest_score['score']:
                        self.highest_scored_mols.remove(lowest_score)
                        self.highest_scored_mols.append({'mol': mol, 'score': new_score})

@hydra.main(config_path='cfgs', config_name='reinforce_trans', version_base=None)
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

    optimizer = reinforce_optimizer(cfg)
    optimizer.optimize(cfg)
    sys.exit(0)
    
if __name__ == '__main__':
    wandb.init(project="llms-materials-rl")
    main()
    sys.exit(0)
    exit()