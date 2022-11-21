import wandb
import hydra
import warnings
warnings.simplefilter("ignore", UserWarning)

from omegaconf import DictConfig

@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if cfg.benchmark == 'gym-classic-control' and cfg.agent_family == 'ppo':
        from workspaces.gym_workspace import PpoGymWorkspace as W
    else:
        raise NotImplementedError

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    if cfg.wandb_log:
        with wandb.init(project=cfg.wandb_project_name, dir=hydra_cfg['runtime']['output_dir'], entity=cfg.wandb_entity, config=dict(cfg),
                        monitor_gym=True, save_code=True):
            wandb.run.name = cfg.wandb_run_name
            workspace = W(cfg)
            workspace.train()
    else:
        workspace = W(cfg)
        workspace.train()
    
if __name__ == '__main__':
    main()