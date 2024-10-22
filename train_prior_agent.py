import os
import sys
import wandb
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig
from optimizer import BaseOptimizer
path_here = os.path.dirname(os.path.realpath(__file__))

from models.reinforce import TransPolicy, RnnPolicy, FcPolicy
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

class prior_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent_name = cfg.agent_name

    def _init(self, cfg):
        if cfg.dataset == 'chembl':
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
            vocab_path = 'data/zinc1m/zinc_' + cfg.rep + '_vocab.txt'
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


        #get prior
        prior_saved_dict = torch.load(os.path.join(path_here, saved_path))
        print('Prior loaded')

        if cfg.model_name == 'char_trans':
            # get agent
            self.agent = TransPolicy(self.vocab, max_dataset_len, cfg.n_heads, cfg.n_embed, cfg.n_layers, dropout=cfg.dropout)
        elif cfg.model_name == 'char_rnn':
            self.agent = RnnPolicy(self.vocab, cfg.rnn_embedding_size, cfg.rnn_hidden_size, cfg.rnn_num_layers).to(self.device)
        elif cfg.model_name == 'char_fc':
            self.agent = FcPolicy(self.vocab, max_dataset_len, cfg.fc_embedding_size, cfg.fc_hidden_size).to(self.device)
        else:
            raise NotImplementedError

        print('Agent class initialised')

        self.agent.to(self.device)
        print('Agent class transferred to cuda memory')

        self.agent.load_save_dict(prior_saved_dict)
        print('Prior weights initialised')

    
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
                obs, episode_lens = self.agent.sample(cfg.batch_size, cfg.max_len, self.device)


            smiles_list = []
            for en_sms in obs:
                sms = self.vocab.decode_padded(en_sms)
                smiles_list.append(sms)
                
            score = np.array(self.predict(smiles_list))

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
                metrics['mean_episode_lens'] = np.mean(episode_lens)
                metrics['max_episode_lens'] = np.max(episode_lens)
                metrics['min_episode_lens'] = np.min(episode_lens)
                wandb.log(metrics)

        print('max training string hit')
        wandb.finish()
        sys.exit(0)

@hydra.main(config_path='cfgs', config_name='prior', version_base=None)
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

    optimizer = prior_optimizer(cfg)
    optimizer.optimize(cfg)
    sys.exit(0)
    
if __name__ == '__main__':
    main()
    sys.exit(0)
    exit()