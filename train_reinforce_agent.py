import os
import time
import wandb
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig
from optimizer import BaseOptimizer
from pathlib import Path
path_here = os.path.dirname(os.path.realpath(__file__))

from models.reinforce import RnnPolicy
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
            self.agent = RnnPolicy(self.vocab, cfg.embedding_size, cfg.hidden_size, cfg.num_layers).to(self.device)
            self.agent.load_save_dict(prior_saved_dict)
        else:
            raise NotImplementedError
        
    def optimize(self, cfg):
        if cfg.wandb_log:
            self.define_wandb_metrics()

        #set device
        self.device = torch.device(cfg.device)

        self._init(cfg)

        patience = 0
        train_steps = 0
        eval_strings = 0
        while eval_strings < cfg.max_strings:
            if len(self) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            with torch.no_grad():
                imag_smiles, imag_smiles_lens = self.agent.sample(cfg.batch_size, cfg.max_len, self.device)

            smiles_list = []
            for en_sms in imag_smiles:
                sms = self.vocab.decode_padded(en_sms)
                smiles_list.append(sms)
                
            score = np.array(self.predict(smiles_list))

            if self.finish:
                print('max oracle hit')
                break 

            # early stopping
            if len(self) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.cfg.patience*2:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            train_steps += 1
            eval_strings += cfg.batch_size
            
            if cfg.wandb_log and train_steps % cfg.train_log_interval == 0:
                metrics = dict()
                metrics['mean_score'] = np.mean(score)
                metrics['max_score'] = np.max(score)
                metrics['min_score'] = np.min(score)
                metrics['mean_len'] = np.mean(imag_smiles_lens)
                metrics['max_len'] = np.max(imag_smiles_lens)
                metrics['min_len'] = np.min(imag_smiles_lens)
                wandb.log(metrics)
        
@hydra.main(config_path='cfgs', config_name='reinforce', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    if cfg.wandb_log:
        project_name = cfg.task + '_' + cfg.target
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg))
        wandb.run.name = cfg.wandb_run_name
    
    set_seed(cfg.seed)
    cfg.output_dir = hydra_cfg['runtime']['output_dir']

    optimizer = reinforce_optimizer(cfg)
    optimizer.optimize(cfg)

if __name__ == '__main__':
    main()