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
        device = self.device 

        obs = torch.zeros((cfg.num_steps, cfg.num_envs) + self.train_envs.single_observation_space.shape).to(device)
        actions = torch.zeros((cfg.num_steps, cfg.num_envs) + self.train_envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        
        start_time = time.time()
        next_obs = torch.Tensor(self.train_envs.reset()).to(device)
        next_done = torch.zeros(cfg.num_envs).to(device)
        num_updates = cfg.num_train_steps // cfg.batch_size

        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * cfg.lr
                self.agent.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, cfg.num_steps):
                self._train_step += 1 * cfg.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, done, info = self.train_envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        self._train_episode += 1
                        print("total episodes: {}, total numsteps: {}, return: {}, cummulative sps: {}".format(self._train_episode, self._train_step, item["episode"]["r"], int(self._train_step / (time.time() - start_time))))
                        if self.cfg.wandb_log:
                            episode_metrics = dict()
                            episode_metrics['episodic_length'] = item["episode"]["l"]
                            episode_metrics['episodic_return'] = item["episode"]["r"]
                            wandb.log(episode_metrics, step=self._train_step)
                        break
            
            train_metrics = dict()
            train_metrics.update(
                self.agent.train(obs, actions, logprobs, rewards, values, next_obs, dones, next_done) 
            )
            train_metrics['SPS'] = int(self._train_step / (time.time() - start_time))

            if self.cfg.wandb_log:
                wandb.log(train_metrics, step=self._train_step)  

        self.train_envs.close()