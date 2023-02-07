import time
import wandb
import hydra
import torch
import random
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np 

from pathlib import Path
from omegaconf import DictConfig

def make_agent(env, device, cfg):    
    obs_dims = np.prod(env.observation_shape)
    num_actions = env.num_actions
    obs_dtype = env.observation_dtype
    action_dtype = np.int32

    env_buffer_size = min(cfg.env_buffer_size, cfg.num_train_steps)

    if cfg.agent == 'sac':
        from sac import SacAgent
        agent = SacAgent(device, obs_dims, num_actions, obs_dtype, action_dtype, env_buffer_size, cfg.gamma, cfg.tau,
                        cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.entropy_coefficient,
                        cfg.hidden_dims, cfg.wandb_log, cfg.agent_log_interval)
    else:
        raise NotImplementedError
    return agent

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
        self.train_env.reset_smiles_batch()
        explore_episode = 0
        
        final_states = np.empty((self.cfg.explore_molecules, self.train_env.observation_shape[0]), dtype=self.train_env.observation_dtype)
        final_next_states = np.empty((self.cfg.explore_molecules, self.train_env.observation_shape[0]), dtype=self.train_env.observation_dtype)
        final_actions = np.empty((self.cfg.explore_molecules, 1), dtype=self.train_env.action_dtype)
        final_rewards = np.zeros((self.cfg.explore_molecules,), dtype=np.float32)
        final_dones = np.zeros((self.cfg.explore_molecules,), dtype=bool)

        while explore_episode < self.cfg.explore_molecules: 
            action = np.random.randint(self.train_env.num_actions)
            next_state, reward, done, info = self.train_env.step(action)
            
            if done:
                final_states[explore_episode] = state 
                final_next_states[explore_episode] = next_state
                final_actions[explore_episode] = action
                final_dones[explore_episode] = done
                explore_episode += 1
                state, done = self.train_env.reset(), False
            else:
                self.agent.env_buffer.push((state, action, reward, next_state, done))
                state = next_state

        reward_start_time = time.time()
        final_rewards, reward_info = self.train_env.get_reward_batch()
        self.train_env.reset_smiles_batch()
        reward_eval_time = time.time() - reward_start_time
        self.agent.env_buffer.push_batch((final_states, final_actions, final_rewards, final_next_states, final_dones), self.cfg.explore_molecules)
        
        print('Total strings = ', len(reward_info['selfies']), 'Unique strings = ', len(set(reward_info['selfies'])), ' Evaluation time = ', reward_eval_time)
        
    def _reset_final_data(self):
        self.train_env.reset_smiles_batch()
        self.final_states = np.empty((self.cfg.parallel_molecules, self.train_env.observation_shape[0]), dtype=self.train_env.observation_dtype)
        self.final_next_states = np.empty((self.cfg.parallel_molecules, self.train_env.observation_shape[0]), dtype=self.train_env.observation_dtype)
        self.final_actions = np.empty((self.cfg.parallel_molecules, 1), dtype=self.train_env.action_dtype)
        self.final_rewards = np.zeros((self.cfg.parallel_molecules,), dtype=np.float32)
        self.final_dones = np.zeros((self.cfg.parallel_molecules,), dtype=bool)
        self._parallel_counter = 0

    def train(self):
        self._explore()
        self._eval()
        self._reset_final_data()
        assert self._train_episode == 0

        # if self.cfg.wandb_log:
        #     wandb.define_metric("episode")
        #     wandb.define_metric("episodic_length", step_metric="episode")
        #     wandb.define_metric("episodic_reward", step_metric="episode")
        #     wandb.define_metric("episodic_sps", step_metric="episode")

        state, done, episode_start_time = self.train_env.reset(), False, time.time()
        for _ in range(1, self.cfg.num_train_steps-self.cfg.explore_steps+1):
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)
            self._train_step += 1
            
            if done:
                self.final_states[self._parallel_counter] = state 
                self.final_next_states[self._parallel_counter] = next_state
                self.final_actions[self._parallel_counter] = action
                self.final_dones[self._parallel_counter] = done
                self._parallel_counter += 1
                self._train_episode += 1
                print("Episode: {}, total numsteps: {}, steps_per_second: {}".format(self._train_episode, self._train_step,info["episode"]["l"]/(time.time() - episode_start_time)))
                state, done, episode_start_time = self.train_env.reset(), False, time.time()
            else:
                self.agent.env_buffer.push((state, action, reward, next_state, done))
                state = next_state

            if self._parallel_counter == self.cfg.parallel_molecules:
                
                reward_start_time = time.time()
                self.final_rewards, reward_info = self.train_env.get_reward_batch()
                reward_eval_time = time.time() - reward_start_time

                print('Total strings = ', len(reward_info['selfies']), 'Unique strings = ', len(set(reward_info['selfies'])), ' Evaluation time = ', reward_eval_time)

                print(self.final_rewards)
                print(np.sort(self.final_rewards))
                
                self.agent.env_buffer.push_batch((self.final_states, self.final_actions, self.final_rewards, self.final_next_states, self.final_dones), self.cfg.parallel_molecules)
                self._reset_final_data()

                if self.cfg.wandb_log:
                    wandb.log({'reward_eval_time' : reward_eval_time}, step = self._train_step)
                exit()

            self.agent.update(self._train_step)

            if self._train_step % self.cfg.eval_episode_interval == 0:
                self._eval()

            if self.cfg.save_snapshot and self._train_step % self.cfg.save_snapshot_interval == 0:
                self.save_snapshot()
                
    def _eval(self):
        steps = 0
        self.eval_env.reset_smiles_batch()
        for _ in range(self.cfg.num_eval_episodes):
            done = False 
            state = self.eval_env.reset()
            while not done:
                action = self.agent.get_action(state, self._train_step, True)
                next_state, _, done ,info = self.eval_env.step(action)
                state = next_state
                
            steps += info["episode"]["l"]

        final_rewards, _ = self.eval_env.get_reward_batch()
        self.eval_env.reset_smiles_batch()
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
        project_name = 'rl4chem'
        with wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=hydra_cfg['runtime']['output_dir']):
            wandb.run.name = cfg.wandb_run_name
            workspace = W(cfg)
            workspace.train()
    else:
        workspace = W(cfg)
        workspace.train()
        
if __name__ == '__main__':
    main()