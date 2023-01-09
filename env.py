import numpy as np
import selfies as sf

from rdkit.Chem import MolFromSmiles
from metrics.metrics import get_logP, get_SA, calc_RingP

class selfies_env(object):
    def __init__(self, episode_length=81, max_selfies_len=81):
        
        # Define alphabet
        self.selfie_alphabet_set = sf.get_semantic_robust_alphabet()
        self.selfie_alphabet_set.add('[nop]')
        self.alphabet_to_idx = {s: i for i, s in enumerate(self.selfie_alphabet_set)}

        # Define action space 
        self.action_space = list(self.selfie_alphabet_set)

        '''Adding phenyl group to alphabet
        '''
        smiles_benzene = "c1ccccc1"
        selfies_benzene = sf.encoder(smiles_benzene)
        self.action_space.append(selfies_benzene)

        self.action_space.sort()
        self.alphabet_length = len(self.selfie_alphabet_set)
        self.action_space_length = len(self.action_space)

        # Set episode length
        self.episode_length = episode_length

        # Set maximum selfies length
        self.max_selfies_length = max_selfies_len

        # Initialize selfie string
        self.molecule_selfie = ''

        # Initialize Step
        self.t = 0

    def onehot_selfies(self, molecule_selfie):
        return np.array(sf.selfies_to_encoding(molecule_selfie, self.alphabet_to_idx, self.max_selfies_length, enc_type='one_hot'), dtype=np.float32)

    def reward(self):
        molecule_smiles = sf.decoder(self.molecule_selfie)
        molecule = MolFromSmiles(molecule_smiles, sanitize=True) # Q_raj : what does sanitize=True do? Will it have any affect in reward calculation?
        
        # normalize scores presumably according to zinc250 dataset. Code from https://github.com/aspuru-guzik-group/GA/blob/paper_results/4.1/random_selfies/random_baseline.py
        reward = (( get_logP(molecule) - 2.4729421499641497 ) / 1.4157879815362406 ) - (( get_SA(molecule) - 3.0470797085649894 ) / 0.830643172314514 ) - (( calc_RingP(molecule) - 0.038131530820234766) / 0.2240274735210179 )
        return reward 
        
    def reset(self):
        # Initialize selfie string
        self.molecule_selfie = ''

        # Initialize Step
        self.t = 0

        return self.molecule_selfie
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length

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

        if done:
            reward = self.reward()
        else:
            reward = 0.0
        
        return self.molecule_selfie, reward, done

if __name__ == '__main__':
    env = selfies_env()
    env.reset()
    done = False
    while not done:
        action = np.random.randint(env.action_space_length)
        state, reward, done = env.step(action)