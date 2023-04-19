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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class dataset_optimizer(BaseOptimizer):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.agent_name = cfg.agent_name

    def _optimize(self, cfg):
        self.oracle.assign_target(cfg)
        device = torch.device(cfg.device)
        
        import pandas as pd
        dataset_path = Path('data/dockstring-dataset.tsv')
        df = pd.read_csv(
            dataset_path, 
            sep="\t", # since our dataset is tab-delimited
            index_col="inchikey",  # index by inchikey
        ) 

        while True:
            smiles_list = np.random.choice(df.smiles.tolist(), size=cfg.batch_size).tolist()
            
            st = time.time()
            affinity_list = self.oracle(smiles_list)
            print(f"Time taken for batch evaluation: {time.time()-st}")

            if self.finish:
                print('max oracle hit')
                break 

            
@hydra.main(config_path='cfgs', config_name='cfg', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    if cfg.wandb_log:
        project_name = 'dockstring_' + cfg.oracle
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=hydra_cfg['runtime']['output_dir'])
        wandb.run.name = cfg.wandb_run_name
    
    set_seed(cfg.seed)
    cfg.output_dir = hydra_cfg['runtime']['output_dir']

    optimizer = dataset_optimizer(cfg)
    optimizer.optimize(cfg, seed=cfg.seed)

if __name__ == '__main__':
    main()