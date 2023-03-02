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
from collections import defaultdict

class Topk():
    def __init__(self):
        self.top25 = []

    def add(self, scores):
        scores = np.sort(np.concatenate([self.top25, scores]))
        self.top25 = scores[-25:]

    def top(self, k):
        assert k <= 25
        return self.top25[-k]

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
    docking_buffer = defaultdict(lambda: None)

    if cfg.agent == 'sac':
        from sac import SacAgent
        agent = SacAgent(device, obs_dims, num_actions, vocab_size, padding_id, cfg.gamma, cfg.tau,
                        cfg.policy_update_interval, cfg.target_update_interval, cfg.lr, cfg.batch_size, cfg.entropy_coefficient,
                        cfg.latent_dims, cfg.hidden_dims, cfg.wandb_log, cfg.agent_log_interval)
    else:
        raise NotImplementedError
    return agent, env_buffer, fresh_env_buffer, docking_buffer

def make_env(cfg):
    print(cfg.id)
    if cfg.id == 'docking':
        from env import docking_env
        return docking_env(cfg), docking_env(cfg)
    else:
        raise NotImplementedError

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collect_random_molecule(env, fresh_env_buffer):
    state, done, t = env.reset(), False, 0
    while not done:
        action = np.random.randint(env.num_actions)
        next_state, reward, done, info = env.step(action)
        fresh_env_buffer.push((state, action, reward, next_state, done, t))
        t += 1
        state = next_state
    return info['episode']

def explore(cfg, train_env, env_buffer, fresh_env_buffer, docking_buffer):
    explore_mols = 0
    while explore_mols < cfg.explore_molecules:
        episode_info = collect_random_molecule(train_env, fresh_env_buffer)

        if docking_buffer[episode_info['smiles']] is not None:
            fresh_env_buffer.remove_last_episode(episode_info['l'])
        else:
            docking_buffer[episode_info['smiles']] = 0
            train_env._add_smiles_to_batch(episode_info['smiles'])
            explore_mols += 1
    
    reward_start_time = time.time()
    parallel_reward_batch, parallel_reward_info = train_env.get_reward_batch()
    reward_eval_time = time.time() - reward_start_time

    #Update main buffer and docking_buffer and reset fresh buffer
    fresh_env_buffer.update_final_rewards(parallel_reward_batch)
    env_buffer.push_fresh_buffer(fresh_env_buffer)
    fresh_env_buffer.reset()
    for i, smiles_string in enumerate(parallel_reward_info['smiles']):
        docking_buffer[smiles_string] = parallel_reward_batch[i]
    
    print('Total strings explored = ', cfg.explore_molecules, ' Reward evaluation time = ', reward_eval_time)
    print(np.sort(parallel_reward_batch))

    return parallel_reward_batch

def collect_molecule(env, agent, fresh_env_buffer):
    state, done, t = env.reset(), False, 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        fresh_env_buffer.push((state, action, reward, next_state, done, t))
        t += 1
        state = next_state
    return info['episode']

def train(cfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    #get train and eval envs
    train_env, eval_env = make_env(cfg)

    #get agent and memory
    agent, env_buffer, fresh_env_buffer, docking_buffer = make_agent(train_env, device, cfg)
    topk = Topk()

    #explore
    docking_scores = explore(cfg, train_env, env_buffer, fresh_env_buffer, docking_buffer)
    topk.add(docking_scores)
    
    #train
    train_step = 0
    molecule_counter = 0
    unique_molecule_counter = 0
    while train_step < cfg.num_train_steps:
        

        episode_info = collect_molecule(train_env, agent, fresh_env_buffer)
        molecule_counter += 1
        if docking_buffer[episode_info['smiles']] is not None:
            fresh_env_buffer.update_last_episode_reward(docking_buffer[episode_info['smiles']])
        else:
            docking_buffer[episode_info['smiles']] = 0
            train_env._add_smiles_to_batch(episode_info['smiles'])
            unique_molecule_counter += 1

        for _ in range(episode_info['l']):
            agent.update(env_buffer, train_step)
            train_step += 1
        
        print('Total strings = ', molecule_counter, 'Unique strings = ', unique_molecule_counter)

        if molecule_counter % cfg.parallel_molecules == 0 and unique_molecule_counter != 0:

            reward_start_time = time.time()
            parallel_reward_batch, parallel_reward_info = train_env.get_reward_batch()
            reward_eval_time = time.time() - reward_start_time
            
            topk.add(parallel_reward_batch)

            #Update main buffer and docking_buffer and reset fresh buffer
            fresh_env_buffer.update_final_rewards(parallel_reward_batch)
            env_buffer.push_fresh_buffer(fresh_env_buffer)
            fresh_env_buffer.reset()
            for i, smiles_string in enumerate(parallel_reward_info['smiles']):
                docking_buffer[smiles_string] = parallel_reward_batch[i]

            print('Evaluation time = ', reward_eval_time)
            print(np.sort(parallel_reward_batch))

            if cfg.wandb_log:
                metrics = dict()
                metrics['reward_eval_time'] = reward_eval_time
                metrics['total_strings'] = molecule_counter
                metrics['unique_strings'] = unique_molecule_counter
                metrics['env_buffer_size'] = len(env_buffer)
                metrics['top1'] = topk.top(1)
                metrics['top5'] = topk.top(5)
                metrics['top25'] = topk.top(25)
                wandb.log(metrics, step=train_step)
                        
    return docking_buffer
        #eval
        #To-do

@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg: DictConfig):

    from train import train
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    if cfg.wandb_log:
        project_name = 'rl4chem_' + cfg.target
        wandb.init(project=project_name, entity=cfg.wandb_entity, config=dict(cfg), dir=hydra_cfg['runtime']['output_dir'])
        wandb.run.name = cfg.wandb_run_name
    
    docking_buffer = train(cfg)
    sorted_docking_buffer = sorted(docking_buffer.items(), key=lambda x:x[1])

    with open(str(hydra_cfg['runtime']['output_dir']) + '/molecules.txt', 'w') as f:
        for (smiles, score) in reversed(sorted_docking_buffer):
            f.write(str(score) + ' || ' + smiles + "\n")

if __name__ == '__main__':
    main()