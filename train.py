import time
import wandb
import hydra
import torch
import random
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np 

from omegaconf import DictConfig
from pathlib import Path

def make_agent(env, device, cfg):    
    if cfg.id == 'selfies':
        obs_dims = np.prod(env.observation_shape)
        num_actions = env.num_actions
        obs_dtype = np.uint8
        action_dtype = np.int32

    elif cfg.id in ['CartPole-v1', 'LunarLander-v2', 'Acrobot-v1', 'MountainCar-v0']:
        obs_dims = np.prod(env.observation_space.shape)
        num_actions = env.action_space.n
        obs_dtype = env.observation_space.dtype
        action_dtype = np.int32

    else:
        raise NotImplementedError

    env_buffer_size = min(cfg.env_buffer_size, cfg.num_train_steps)

    if cfg.agent == 'sac':
        from sac import SacAgent
        agent = SacAgent(device, obs_dims, num_actions, obs_dtype, action_dtype, env_buffer_size, cfg.gamma, cfg.tau,
                        cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.entropy_coefficient,
                        cfg.hidden_dims, cfg.wandb_log, cfg.log_interval)
    else:
        raise NotImplementedError
    
    return agent

def make_env(cfg):
    print(cfg.id)
    if cfg.id == 'selfies':
        from env import selfies_env
        return selfies_env(cfg.max_selfie_length, cfg.max_selfie_length, target=cfg.target), selfies_env(cfg.max_selfie_length, cfg.max_selfie_length, target=cfg.target)
    elif cfg.id in ['CartPole-v1', 'LunarLander-v2', 'Acrobot-v1', 'MountainCar-v0']:
        import gym  
        def make_env(cfg):
            env = gym.make(cfg.id) 
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed=cfg.seed)
            env.observation_space.seed(cfg.seed)
            env.action_space.seed(cfg.seed)
            return env
        return make_env(cfg), make_env(cfg)
    else:
        raise NotImplementedError

class RewardshapedWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        if self.cfg.save_snapshot:
            self.checkpoint_path = self.work_dir / 'checkpoints'
            self.checkpoint_path.mkdir(exist_ok=True)

        self.set_seed()
        self.device = torch.device(cfg.device)
        self.train_env, self.eval_env = make_env(self.cfg)
        self.agent = make_agent(self.train_env, self.device, self.cfg)
        self._train_step = 0
        self._train_episode = 0
        self._best_eval_returns = -np.inf
        self._best_train_returns = -np.inf            

    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def _explore(self):
        print('random exploration begins')
        state, done = self.train_env.reset(), False
        explore_episode = 0
        for _ in range(1, self.cfg.explore_steps):
            
            if self.cfg.id == 'selfies':
                action = np.random.randint(self.train_env.num_actions)
            elif self.cfg.id in ['CartPole-v1', 'LunarLander-v2', 'Acrobot-v1', 'MountainCar-v0']:
                action = self.train_env.action_space.sample()
            else:
                raise NotImplementedError

            next_state, reward, done, info = self.train_env.step(action)

            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            if done:
                explore_episode += 1
                state, done = self.train_env.reset(), False
            else:
                state = next_state
        print('random exploration complete')

    def _collect_episode(self):
        episode_metrics = dict()
        state, done = self.train_env.reset(), False
        states = np.empty((self.cfg.max_selfie_length + 1, np.prod(self.train_env.observation_shape)), dtype=self.agent.env_buffer.obs_dtype) 
        actions = np.empty((self.cfg.max_selfie_length, 1), dtype=self.agent.env_buffer.action_dtype)
        rewards = np.zeros((self.cfg.max_selfie_length,), dtype=np.float32) 
        terminals = np.zeros((self.cfg.max_selfie_length,), dtype=bool)
        
        step = 0
        states[step] = state 
        while not done:
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)

            if not done: 
                assert reward == 0
            if done:
                rewards = rewards + reward
                terminals[step] = False if info.get("TimeLimit.truncated", False) else done

            states[step+1] = next_state
            actions[step] = action
            step += 1
            state = next_state

        assert step==info["episode"]["l"]
        self.agent.env_buffer.push_sequence((states[:-1], actions, rewards, states[1:], terminals), info["episode"]["l"])
        episode_metrics['episodic_return'] = info["episode"]["r"]            
        episode_metrics['episodic_length'] = info["episode"]["l"]
        episode_metrics.update(info['episode_logs'])
        episode_metrics['env_buffer_length'] = len(self.agent.env_buffer)
        
        if self.cfg.id in ['selfies'] and info["episode"]["r"] > self._best_train_returns:
            self._best_train_returns = info["episode"]["r"]
            print('Total numsteps: {}, smile: {}, pLogP score: {} \n'.format(self._train_step, info['smiles'], round(episode_metrics['episodic_return'], 2)))
        else:
            print("Episode: {}, total numsteps: {}, return: {}".format(self._train_episode, self._train_step, round(episode_metrics['episodic_return'], 2)))
        return episode_metrics

    def train(self):
        self._explore()
        self._eval()

        while self._train_step < self.cfg.num_train_steps-self.cfg.explore_steps+1:
    
            episode_start_time = time.time()
            episode_metrics = dict()
            episode_metrics.update(self._collect_episode())
            
            for _ in range(episode_metrics["episodic_length"]):

                self.agent.update(self._train_step)
                self._train_step += 1

                if self._train_step % self.cfg.eval_episode_interval == 0:
                        self._eval()

                if self.cfg.save_snapshot and self._train_step % self.cfg.save_snapshot_interval == 0:
                    self.save_snapshot()
            
            if self.cfg.wandb_log:
                episode_metrics['steps_per_second'] = episode_metrics["episodic_length"]/(time.time() - episode_start_time)
                wandb.log(episode_metrics, step=self._train_step)
                            
    def _eval(self):
        returns = 0 
        steps = 0
        for _ in range(self.cfg.num_eval_episodes):
            done = False 
            state = self.eval_env.reset()
            while not done:
                action = self.agent.get_action(state, self._train_step, True)
                next_state, _, done ,info = self.eval_env.step(action)
                state = next_state
                
            returns += info["episode"]["r"]
            steps += info["episode"]["l"]
            
        eval_metrics = dict()
        eval_metrics['eval_episodic_return'] = returns/self.cfg.num_eval_episodes
        eval_metrics['eval_episodic_length'] = steps/self.cfg.num_eval_episodes

        print("Episode: {}, total numsteps: {}, average Evaluation return: {}".format(self._train_episode, self._train_step, round(eval_metrics['eval_episodic_return'], 2)))

        if self.cfg.save_snapshot and returns/self.cfg.num_eval_episodes >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = returns/self.cfg.num_eval_episodes

        if self.cfg.wandb_log:
            wandb.log(eval_metrics, step = self._train_step)

    def save_snapshot(self, best=False):
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(self._train_step)+'.pt')
        save_dict = self.agent.get_save_dict()
        torch.save(save_dict, snapshot)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        if self.cfg.save_snapshot:
            self.checkpoint_path = self.work_dir / 'checkpoints'
            self.checkpoint_path.mkdir(exist_ok=True)

        self.set_seed()
        self.device = torch.device(cfg.device)
        self.train_env, self.eval_env = make_env(self.cfg)
        self.agent = make_agent(self.train_env, self.device, self.cfg)
        self._train_step = 0
        self._train_episode = 0
        self._best_eval_returns = -np.inf
        self._best_train_returns = -np.inf

    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def _explore(self):
        print('random exploration begins')
        state, done = self.train_env.reset(), False
        explore_episode = 0
        for _ in range(1, self.cfg.explore_steps):
            
            if self.cfg.id == 'selfies':
                action = np.random.randint(self.train_env.num_actions)
            elif self.cfg.id in ['CartPole-v1', 'LunarLander-v2', 'Acrobot-v1', 'MountainCar-v0']:
                action = self.train_env.action_space.sample()
            else:
                raise NotImplementedError

            next_state, reward, done, info = self.train_env.step(action)

            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            if done:
                explore_episode += 1
                state, done = self.train_env.reset(), False
            else:
                state = next_state
        print('random exploration complete')

    def train(self):
        self._explore()
        self._eval()

        state, done, episode_start_time = self.train_env.reset(), False, time.time()
        
        for _ in range(1, self.cfg.num_train_steps-self.cfg.explore_steps+1):
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)
            self._train_step += 1
            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            self.agent.update(self._train_step)

            if self._train_step % self.cfg.eval_episode_interval == 0:
                    self._eval()

            if self.cfg.save_snapshot and self._train_step % self.cfg.save_snapshot_interval == 0:
                self.save_snapshot()
        
            if done:
                self._train_episode += 1
                episode_metrics = dict()
                episode_metrics['episodic_return'] = info["episode"]["r"]
                    
                if self.cfg.wandb_log:
                    episode_metrics['episodic_length'] = info["episode"]["l"]
                    episode_metrics['steps_per_second'] = info["episode"]["l"]/(time.time() - episode_start_time)
                    if self.cfg.id in ['selfies']:
                        episode_metrics.update(info['episode_logs'])
                    episode_metrics['env_buffer_length'] = len(self.agent.env_buffer)
                    wandb.log(episode_metrics, step=self._train_step)
                
                if self.cfg.id in ['selfies'] and info["episode"]["r"] > self._best_train_returns:
                    self._best_train_returns = info["episode"]["r"]
                    print('Total numsteps: {}, smile: {}, pLogP score: {} \n'.format(self._train_step, info['smiles'], round(episode_metrics['episodic_return'], 2)))
                else:
                    print("Episode: {}, total numsteps: {}, return: {}".format(self._train_episode, self._train_step, round(episode_metrics['episodic_return'], 2)))
                    
                state, done, episode_start_time = self.train_env.reset(), False, time.time()
            else:
                state = next_state
                
    def _eval(self):
        returns = 0 
        steps = 0
        for _ in range(self.cfg.num_eval_episodes):
            done = False 
            state = self.eval_env.reset()
            while not done:
                action = self.agent.get_action(state, self._train_step, True)
                next_state, _, done ,info = self.eval_env.step(action)
                state = next_state
                
            returns += info["episode"]["r"]
            steps += info["episode"]["l"]
            
        eval_metrics = dict()
        eval_metrics['eval_episodic_return'] = returns/self.cfg.num_eval_episodes
        eval_metrics['eval_episodic_length'] = steps/self.cfg.num_eval_episodes

        print("Episode: {}, total numsteps: {}, average Evaluation return: {}".format(self._train_episode, self._train_step, round(eval_metrics['eval_episodic_return'], 2)))

        if self.cfg.save_snapshot and returns/self.cfg.num_eval_episodes >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = returns/self.cfg.num_eval_episodes

        if self.cfg.wandb_log:
            wandb.log(eval_metrics, step = self._train_step)

    def save_snapshot(self, best=False):
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(self._train_step)+'.pt')
        save_dict = self.agent.get_save_dict()
        torch.save(save_dict, snapshot)

@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if cfg.shape:
        from train import RewardshapedWorkspace as W
    else:
        from train import Workspace as W
        
    if cfg.wandb_log:
        project_name = 'rd4chem'
        with wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg)):
            wandb.run.name = cfg.wandb_run_name
            workspace = W(cfg)
            workspace.train()
    else:
        workspace = W(cfg)
        workspace.train()
        
if __name__ == '__main__':
    main()