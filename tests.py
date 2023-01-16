import hydra
import torch
import random
import warnings
import numpy as np 
warnings.simplefilter("ignore", UserWarning)

from pathlib import Path
from tabulate import tabulate
from omegaconf import DictConfig

from env_probe import level_one_env, level_two_env, level_three_env, level_four_env, level_five_env

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        if self.cfg.save_snapshot:
            self.checkpoint_path = self.work_dir / 'checkpoints'
            self.checkpoint_path.mkdir(exist_ok=True)

        self.device = torch.device(cfg.device)
        self._train_step = 0
        self._train_episode = 0
        self._best_eval_returns = -np.inf

        # self._probe_level_one(cfg)    # One action, zero observation, one timestep long, +1 reward every timestep:
        # self._probe_level_two(cfg)    # One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward every time
        # self._probe_level_three(cfg)  # One action, zero-then-one observation, two timesteps long, +1 reward at the end
        # self._probe_level_four(cfg)   # Two actions, zero observation, one timestep long, action-dependent +1/-1 reward
        # self._probe_level_five(cfg)   # Two actions, random +1/-1 observation, one timestep long, action-and-obs dependent +1/-1 reward

    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

    def _explore(self):
        state, done = self.train_env.reset(), False
        explore_episode = 0
        for _ in range(1, 10000):
            action = np.random.randint(self.train_env.action_space_length)

            next_state, reward, done, info = self.train_env.step(action)
            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))
            if done:
                explore_episode += 1
                state, done = self.train_env.reset(), False
            else:
                state = next_state

    def _probe_level_five(self, cfg):
        self.train_env = level_five_env()
        obs_dims = self.train_env.observation_length
        num_actions = self.train_env.action_space_length
        obs_dtype = np.int32
        action_dtype = np.int32

        env_buffer_size = int(min(cfg.env_buffer_size, cfg.num_train_steps))
        
        if cfg.agent == 'sac':
            from sac import SacAgent
            self.agent = SacAgent(self.device, obs_dims, num_actions, obs_dtype, action_dtype, env_buffer_size, cfg.gamma, cfg.tau,
                            cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.entropy_coefficient,
                            cfg.hidden_dims, False, 1)
        else:
            raise NotImplementedError       

        self._explore()

        state, done = self.train_env.reset(), False
        
        for _ in range(5000):
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)
            self._train_step += 1
            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            self.agent.update(self._train_step)

            if done:
                self._train_episode += 1
                state, done = self.train_env.reset(), False
            else:
                state = next_state

            if self._train_step == 1 or self._train_step % 500 == 0:
                observations = self.train_env.observation_space
                with torch.no_grad():
                    observations = torch.FloatTensor(observations).to(self.device) 
                    Q1, Q2 = self.agent.critic(observations)
                    targ_Q1, targ_Q2 = self.agent.critic_target(observations)
                    dist = self.agent.actor(observations)
 
                    column_names1 = ['values', 'state 1-action 1', 'state 1-action 2', 'state 2-action 1', 'state 2-action 2']
                    target_values1 = ['optimal', 1.0, -1.0, -1.0, 1.0]
                    observed_values1_1 = ['agent\'s Q1', Q1.cpu().numpy()[0,0], Q1.cpu().numpy()[0,1], Q1.cpu().numpy()[1,0], Q1.cpu().numpy()[0,0], Q1.cpu().numpy()[1,1]]
                    observed_values1_2 = ['agent\'s Q2', Q2.cpu().numpy()[0,0], Q2.cpu().numpy()[0,1], Q2.cpu().numpy()[1,0], Q2.cpu().numpy()[0,0], Q2.cpu().numpy()[1,1]]

                    table1 = [column_names1, target_values1, observed_values1_1, observed_values1_2]

                    column_names2 = ['action probabilities', 'state 1-action 1', 'state 1-action 2', 'state 2-action 1', 'state 2-action 2']
                    target_values2 = ['optimal', 1.0, 0.0, 0.0, 1.0]
                    observed_values2 = ['agent', dist.probs.cpu().numpy()[0,0], dist.probs.cpu().numpy()[0,1], dist.probs.cpu().numpy()[1,0], dist.probs.cpu().numpy()[1,1]]

                    table2 = [column_names2, target_values2, observed_values2]


                    print('training step = ', self._train_step)
                    print(tabulate(table1, headers='firstrow', tablefmt='fancy_grid'))
                    print(tabulate(table2, headers='firstrow', tablefmt='fancy_grid'))
                    print('===================================================================================================================================================')


    def _probe_level_four(self, cfg):
        self.train_env = level_four_env()
        obs_dims = self.train_env.observation_length
        num_actions = self.train_env.action_space_length
        obs_dtype = np.int32
        action_dtype = np.int32

        env_buffer_size = int(min(cfg.env_buffer_size, cfg.num_train_steps))
        
        if cfg.agent == 'sac':
            from sac import SacAgent
            self.agent = SacAgent(self.device, obs_dims, num_actions, obs_dtype, action_dtype, env_buffer_size, cfg.gamma, cfg.tau,
                            cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.entropy_coefficient,
                            cfg.hidden_dims, False, 1)
        else:
            raise NotImplementedError       

        self._explore()

        state, done = self.train_env.reset(), False
        
        for _ in range(5000):
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)
            self._train_step += 1
            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            self.agent.update(self._train_step)

            if done:
                self._train_episode += 1
                state, done = self.train_env.reset(), False
            else:
                state = next_state

            if self._train_step == 1 or self._train_step % 500 == 0:
                observations = self.train_env.observation_space
                with torch.no_grad():
                    observations = torch.FloatTensor(observations).to(self.device) 
                    Q1, Q2 = self.agent.critic(observations)
                    targ_Q1, targ_Q2 = self.agent.critic_target(observations)
                    dist = self.agent.actor(observations)
 
                    column_names1 = ['values', 'action 1 Q1', 'action 1 Q2', 'action 1 target Q1', 'action 1 target Q2', 'action 2 Q1', 'action 2 Q2', 'action 2 target Q1', 'action 2 target Q2']
                    target_values1 = ['optimal', -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
                    observed_values1 = ['agent', Q1.cpu().numpy()[0,0], Q2.cpu().numpy()[0,0], targ_Q1.cpu().numpy()[0,0], targ_Q2.cpu().numpy()[0,0], Q1.cpu().numpy()[0,-1], Q2.cpu().numpy()[0,-1], targ_Q1.cpu().numpy()[0,-1], targ_Q2.cpu().numpy()[0,-1]]

                    table1 = [column_names1, target_values1, observed_values1]

                    column_names2 = ['action probabilities', 'action 1', 'action 2']
                    target_values2 = ['optimal', 0.0, 1.0]
                    observed_values2 = ['agent', dist.probs.cpu().numpy()[0,0], dist.probs.cpu().numpy()[0,1]]

                    table2 = [column_names2, target_values2, observed_values2]


                    print('training step = ', self._train_step)
                    print(tabulate(table1, headers='firstrow', tablefmt='fancy_grid'))
                    print(tabulate(table2, headers='firstrow', tablefmt='fancy_grid'))
                    print('\n')
    
    def _probe_level_three(self, cfg):
        self.train_env = level_three_env()
        obs_dims = self.train_env.observation_length
        num_actions = self.train_env.action_space_length
        obs_dtype = np.int32
        action_dtype = np.int32

        env_buffer_size = int(min(cfg.env_buffer_size, cfg.num_train_steps))
        
        if cfg.agent == 'sac':
            from sac import SacAgent
            self.agent = SacAgent(self.device, obs_dims, num_actions, obs_dtype, action_dtype, env_buffer_size, cfg.gamma, cfg.tau,
                            cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.target_entropy_ratio,
                            cfg.hidden_dims, False, 1)
        else:
            raise NotImplementedError       

        self._explore()

        state, done = self.train_env.reset(), False
        
        for _ in range(5000):
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)
            self._train_step += 1
            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            self.agent.update(self._train_step)

            if done:
                self._train_episode += 1
                state, done = self.train_env.reset(), False
            else:
                state = next_state

            if self._train_step % 500 == 0:
                observations = self.train_env.observation_space
                with torch.no_grad():
                    observations = torch.FloatTensor(observations).to(self.device) 
                    Q1, Q2 = self.agent.critic(observations)
                    targ_Q1, targ_Q2 = self.agent.critic_target(observations)
                
                    column_names = ['values', 'state 1 Q1', 'state 1 Q2', 'state 1 target Q1', 'state 1 target Q2', 'state 2 Q1', 'state 2 Q2', 'state 2 target Q1', 'state 2 target Q2']
                    target_values = ['optimal', cfg.gamma, cfg.gamma, cfg.gamma, cfg.gamma, 1.0, 1.0, 1.0, 1.0]
                    observed_values = ['agent', Q1.cpu().numpy()[0], Q2.cpu().numpy()[0], targ_Q1.cpu().numpy()[0], targ_Q2.cpu().numpy()[0], Q1.cpu().numpy()[-1], Q2.cpu().numpy()[-1], targ_Q1.cpu().numpy()[-1], targ_Q2.cpu().numpy()[-1]]

                    table = [column_names, target_values, observed_values]
                    print('training step = ', self._train_step)
                    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    def _probe_level_two(self, cfg):
        self.train_env = level_two_env()
        obs_dims = self.train_env.observation_length
        num_actions = self.train_env.action_space_length
        obs_dtype = np.int32
        action_dtype = np.int32

        env_buffer_size = int(min(cfg.env_buffer_size, cfg.num_train_steps))
        
        if cfg.agent == 'sac':
            from sac import SacAgent
            self.agent = SacAgent(self.device, obs_dims, num_actions, obs_dtype, action_dtype, env_buffer_size, cfg.gamma, cfg.tau,
                            cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.target_entropy_ratio,
                            cfg.hidden_dims, False, 1)
        else:
            raise NotImplementedError       

        self._explore()

        state, done = self.train_env.reset(), False
        
        for _ in range(5000):
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)
            self._train_step += 1
            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            self.agent.update(self._train_step)

            if done:
                self._train_episode += 1
                state, done = self.train_env.reset(), False
            else:
                state = next_state

            if self._train_step % 500 == 0:
                observations = self.train_env.observation_space
                with torch.no_grad():
                    observations = torch.FloatTensor(observations).to(self.device) 
                    Q1, Q2 = self.agent.critic(observations)
                    targ_Q1, targ_Q2 = self.agent.critic_target(observations)
                
                    column_names = ['values', 'state 1 Q1', 'state 1 Q2', 'state 1 target Q1', 'state 1 target Q2', 'state 2 Q1', 'state 2 Q2', 'state 2 target Q1', 'state 2 target Q2']
                    target_values = ['optimal', -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
                    observed_values = ['agent', Q1.cpu().numpy()[0], Q2.cpu().numpy()[0], targ_Q1.cpu().numpy()[0], targ_Q2.cpu().numpy()[0], Q1.cpu().numpy()[-1], Q2.cpu().numpy()[-1], targ_Q1.cpu().numpy()[-1], targ_Q2.cpu().numpy()[-1]]

                    table = [column_names, target_values, observed_values]
                    print('training step = ', self._train_step)
                    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    def _probe_level_one(self, cfg):
        self.train_env = level_one_env()
        obs_dims = self.train_env.observation_length
        num_actions = self.train_env.action_space_length
        obs_dtype = np.int32
        action_dtype = np.int32

        env_buffer_size = int(min(cfg.env_buffer_size, cfg.num_train_steps))
        
        if cfg.agent == 'sac':
            from sac import SacAgent
            self.agent = SacAgent(self.device, obs_dims, num_actions, obs_dtype, action_dtype, env_buffer_size, cfg.gamma, cfg.tau,
                            cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.target_entropy_ratio,
                            cfg.hidden_dims, False, 1)
        else:
            raise NotImplementedError       

        self._explore()

        state, done = self.train_env.reset(), False
        
        for _ in range(5000):
            action = self.agent.get_action(state, self._train_step)
            next_state, reward, done, info = self.train_env.step(action)
            self._train_step += 1
            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            self.agent.update(self._train_step)

            if done:
                self._train_episode += 1
                state, done = self.train_env.reset(), False
            else:
                state = next_state

            if self._train_step % 500 == 0:
                observations = self.train_env.observation_space
                with torch.no_grad():
                    observations = torch.FloatTensor(observations).to(self.device) 
                    Q1, Q2 = self.agent.critic(observations)
                    targ_Q1, targ_Q2 = self.agent.critic_target(observations)

                    column_names = ['values', 'Q1', 'Q2', 'target Q1', 'target Q2']
                    target_values = ['optimal', 1.0, 1.0, 1.0, 1.0]
                    observed_values = ['agent', Q1.cpu().numpy()[0], Q2.cpu().numpy()[0], targ_Q1.cpu().numpy()[0], targ_Q2.cpu().numpy()[0]]

                    table = [column_names, target_values, observed_values]
                    print('training step = ', self._train_step)
                    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
                
@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    from tests import Workspace as W
    W(cfg)
        
if __name__ == '__main__':
    main()