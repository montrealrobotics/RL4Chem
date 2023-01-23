import numpy as np
import selfies as sf

from rdkit import Chem
from collections import defaultdict, deque
from metrics.metrics import get_penalized_logp_reward, get_QED_reward, steric_strain_filter, zinc_molecule_filter

class selfies_env(object):
    '''This environment is build assuming selfies version 2.1.1
    To-do
    1) Register as an official gym environment
    '''
    def __init__(self, episode_length=72, max_selfies_len=72, target='plogp'):
        # max selfies length was chosen because it is the maximum selfie length in the zinc250 dataset (selfies version 2.1.1)
        # Set maximum selfies length
        self.max_selfies_length = max_selfies_len

        # Set episode length
        self.episode_length = episode_length

        # Set target property
        self.target = target
        if self.target == 'plogp':
            self.property_fn = get_penalized_logp_reward
        elif self.target == 'qed':
            self.property_fn = get_QED_reward
        else:
            raise NotImplementedError

        # Define action space 
        self.selfie_alphabet_set = sf.get_semantic_robust_alphabet()
        
        self.selfie_alphabet_set.add('[nop]')
        self.selfie_alphabet = sorted(list(self.selfie_alphabet_set))
        self.alphabet_to_idx = {s: i for i, s in enumerate(self.selfie_alphabet)}

        self.action_space = list(self.selfie_alphabet_set) # add a termination action ??
        self.action_space.sort()
        self.alphabet_length = len(self.selfie_alphabet_set)
        self.action_space_length = len(self.action_space)
        
        # Set observation and action space shape
        self.observation_shape = (self.max_selfies_length*self.alphabet_length,)
        self.num_actions = self.action_space_length

        # Initialize selfie string
        self.molecule_selfie = ''

        # Initialize Step
        self.t = 0

    def onehot_selfies(self, molecule_selfie):
        return np.array(sf.selfies_to_encoding(molecule_selfie, self.alphabet_to_idx, self.max_selfies_length, enc_type='one_hot'), dtype=np.uint8).flatten()
        
    def reset(self):
        # Initialize selfie string
        self.molecule_selfie = ''

        # Initialize Step
        self.t = 0

        return self.onehot_selfies(self.molecule_selfie)
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length
        info = defaultdict(dict)

        action_selfie = self.action_space[action]
        new_molecule_selfie = self.molecule_selfie + action_selfie
        new_molecule_len = sf.len_selfies(new_molecule_selfie)

        self.t += 1
        
        if new_molecule_len > self.max_selfies_length:
            done = True 
        elif new_molecule_len == self.max_selfies_length:
            done = True
            self.molecule_selfie = new_molecule_selfie
        else:
            done = False
            self.molecule_selfie = new_molecule_selfie

        if not done and self.t >= self.episode_length:
            done = True

        if done:
            molecule_smiles = sf.decoder(self.molecule_selfie)
            molecule = Chem.MolFromSmiles(molecule_smiles, sanitize=True)
            reward, reward_info = self.property_fn(molecule)
            info["episode_logs"].update(reward_info)
            info["episode"]["r"] = reward
            info["episode"]["l"] = self.t
            info['smiles'] = molecule_smiles
        else:
            # reward = -1
            reward = 0

        return self.onehot_selfies(self.molecule_selfie), reward, done, info

if __name__ == '__main__':
    env = selfies_env()
    # for i in range(1):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = np.random.randint(env.action_space_length)
    #         state, reward, done, info = env.step(action)
    #         print(info)