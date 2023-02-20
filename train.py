import time
import wandb
import hydra
import torch
import random
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import utils

from pathlib import Path
from omegaconf import DictConfig

def make_agent(env, device, cfg):    
    obs_dims = np.prod(env.observation_shape)
    num_actions = env.num_actions
    vocab_size = env.alphabet_length
    padding_id = env.alphabet_to_idx['[nop]']
    obs_dtype = env.observation_dtype
    action_dtype = np.int32

    env_buffer_size = min(cfg.env_buffer_size, cfg.num_train_steps)

    env_buffer = utils.ReplayMemory(env_buffer_size, obs_dims, obs_dtype, action_dtype)
    fresh_env_buffer = utils.FreshReplayMemory(cfg.parallel_molecules, env.episode_length, obs_dims, obs_dtype, action_dtype)

    if cfg.agent == 'sac':
        from sac import SacAgent
        agent = SacAgent(device, obs_dims, num_actions, vocab_size, padding_id, cfg.gamma, cfg.tau,
                        cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.entropy_coefficient,
                        cfg.latent_dims, cfg.hidden_dims, cfg.wandb_log, cfg.agent_log_interval)
    else:
        raise NotImplementedError
    return agent, env_buffer, fresh_env_buffer

def make_env(cfg):
    print(cfg.id)
    if cfg.id == 'docking':
        from env import docking_env
        return docking_env(cfg), docking_env(cfg)
    else:
        raise NotImplementedError

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
        self.agent, self.env_buffer, self.fresh_env_buffer = make_agent(self.train_env, self.device, self.cfg)
        self.current_reward_batch = np.zeros((cfg.parallel_molecules,), dtype=np.float32)
        self.current_reward_info = dict()
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
        print('random exploration of ', self.cfg.parallel_molecules, ' number of molecules begins')
        explore_steps = self.cfg.parallel_molecules * self.train_env.episode_length

        state, done = self.train_env.reset(), False
        for _ in range(explore_steps): 
            action = np.random.randint(self.train_env.num_actions)
            next_state, reward, done, info = self.train_env.step(action)
            
            self.fresh_env_buffer.push((state, action, reward, next_state, done))
            
            if done:
                state, done = self.train_env.reset(), False
            else:
                state = next_state
        
        reward_start_time = time.time()
        self.current_reward_batch, self.current_reward_info = self.train_env.get_reward_batch()
        reward_eval_time = time.time() - reward_start_time
        self.fresh_env_buffer.update_final_rewards(self.current_reward_batch)
        self.env_buffer.push_fresh_buffer(self.fresh_env_buffer)
        self.fresh_env_buffer.reset()
        print('Total strings = ', len(self.current_reward_info['selfies']), 'Unique strings = ', len(set(self.current_reward_info['selfies'])), ' Evaluation time = ', reward_eval_time)
        print(np.sort(self.current_reward_batch))

    def train(self):
        self._eval()
        self._explore()

        parallel_counter = 0
        state, done, episode_start_time, episode_metrics = self.train_env.reset(), False, time.time(), dict()
        
        for _ in range(1, self.cfg.num_train_steps):
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)
            self.fresh_env_buffer.push((state, action, reward, next_state, done))
            self._train_step += 1
            
            if done:
                self._train_episode += 1
                print("Episode: {}, total numsteps: {}".format(self._train_episode, self._train_step)) 
                state, done, episode_start_time = self.train_env.reset(), False, time.time()
                if self.cfg.wandb_log:
                    episode_metrics['episodic_length'] = info["episode"]["l"]
                    episode_metrics['steps_per_second'] = info["episode"]["l"]/(time.time() - episode_start_time)
                    episode_metrics['env_buffer_length'] = len(self.env_buffer)
                    episode_metrics['episodic_reward'] = self.current_reward_batch[parallel_counter]
                    episode_metrics['episodic_selfies_len'] = self.current_reward_info['len_selfies'][parallel_counter]
                    wandb.log(episode_metrics, step=self._train_step)
                parallel_counter += 1
            else:
                state = next_state

            self.agent.update(self.env_buffer, self._train_step)

            if self._train_step % self.cfg.eval_episode_interval == 0:
                self._eval()

            if self.cfg.save_snapshot and self._train_step % self.cfg.save_snapshot_interval == 0:
                self.save_snapshot()

            if parallel_counter == self.cfg.parallel_molecules:
                reward_start_time = time.time()
                self.current_reward_batch, self.current_reward_info = self.train_env.get_reward_batch()
                reward_eval_time = time.time() - reward_start_time
                self.fresh_env_buffer.update_final_rewards(self.current_reward_batch)
                self.env_buffer.push_fresh_buffer(self.fresh_env_buffer)
                self.fresh_env_buffer.reset()

                unique_strings = len(set(self.current_reward_info['selfies']))
                print('Total strings = ', len(self.current_reward_info['selfies']), 'Unique strings = ', unique_strings, ' Evaluation time = ', reward_eval_time)
                print(np.sort(self.current_reward_batch))
                best_idx = np.argmax(self.current_reward_batch)
                print(self.current_reward_info['smiles'][best_idx])
                
                if self.cfg.wandb_log:
                    wandb.log({'reward_eval_time' : reward_eval_time, 
                                'unique strings': unique_strings}, step = self._train_step)
                parallel_counter = 0
                
    def _eval(self):
        steps = 0
        for _ in range(self.cfg.num_eval_episodes):
            done = False 
            state = self.eval_env.reset()
            while not done:
                action = self.agent.get_action(state, self._train_step, True)
                next_state, _, done ,info = self.eval_env.step(action)
                state = next_state
                
            steps += info["episode"]["l"]

        final_rewards, _ = self.eval_env.get_reward_batch()
        eval_metrics = dict()
        eval_metrics['eval_episodic_return'] = sum(final_rewards)/self.cfg.num_eval_episodes
        eval_metrics['eval_episodic_length'] = steps/self.cfg.num_eval_episodes

        print("Episode: {}, total numsteps: {}, average Evaluation return: {}".format(self._train_episode, self._train_step, round(eval_metrics['eval_episodic_return'], 2)))

        if self.cfg.save_snapshot and sum(final_rewards)/self.cfg.num_eval_episodes >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = sum(final_rewards)/self.cfg.num_eval_episodes

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

    from train import Workspace as W
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    if cfg.wandb_log:
        project_name = 'rl4chem_' + cfg.target
        with wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=hydra_cfg['runtime']['output_dir']):
            wandb.run.name = cfg.wandb_run_name
            workspace = W(cfg)
            workspace.train()
    else:
        workspace = W(cfg)
        workspace.train()
        
if __name__ == '__main__':
    main()