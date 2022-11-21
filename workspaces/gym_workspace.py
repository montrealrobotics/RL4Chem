import time
import torch
import wandb
import random
import numpy as np

from pathlib import Path 
from utils import CartPolePOMDPWrapper

def make_env(cfg):
    if 'gym' in cfg.benchmark:
        import gym
        def get_gymenv(cfg):
            def thunk():
                env = gym.make(cfg.id) 
                if cfg.pomdp:
                    assert cfg.id == 'CartPole-v1'
                    env = CartPolePOMDPWrapper(env)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env.seed(seed=cfg.seed)
                env.observation_space.seed(cfg.seed)
                env.action_space.seed(cfg.seed)
                return env 
            return thunk

        return gym.vector.SyncVectorEnv([get_gymenv(cfg) for i in range(cfg.num_envs)])
    
    else:
        raise NotImplementedError

def make_agent(envs, cfg, device):
    if cfg.agent == 'ppo':
        from agents.ppo import PpoAgent
        agent = PpoAgent(envs, cfg, device)
    elif cfg.agent == 'ppo_lstm':
        from agents.ppo_lstm import PpoLstmAgent
        agent = PpoLstmAgent(envs, cfg, device)                   
    else:
        raise NotImplementedError

    return agent

class PpoGymWorkspace:
    def __init__(self, cfg):
        assert 'ppo' in cfg.agent
        self.work_dir = Path.cwd()
        
        if cfg.save_snapshot:
            self.checkpoint_path = self.work_dir / 'checkpoints'
            self.checkpoint_path.mkdir(exist_ok=True)
        
        self.device = torch.device(cfg.device)
        self.set_seed(cfg)
        self.train_envs = make_env(cfg)
        self.agent = make_agent(self.train_envs, cfg, self.device)
        cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
        cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
        self.cfg = cfg

        self._train_step = 0
        self._train_episode = 0
        self._best_eval_returns = -np.inf

    def set_seed(self, cfg):
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

    def train(self):
        cfg = self.cfg
        start_time = time.time()
        num_updates = cfg.num_train_steps // cfg.batch_size

        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * cfg.lr
                self.agent.optimizer.param_groups[0]["lr"] = lrnow

            self._train_step, self._train_episode =  self.agent.collect_data(self.train_envs, self._train_step, self._train_episode, start_time)

            train_metrics = dict()
            train_metrics.update(
                self.agent.train(self.train_envs) 
            )
            train_metrics['SPS'] = int(self._train_step / (time.time() - start_time))

            if self.cfg.wandb_log:
                wandb.log(train_metrics, step=self._train_step)  

        self.train_envs.close()