import numpy as np
import selfies as sf

from collections import defaultdict
from rdkit.Chem import MolFromSmiles, Draw
from metrics.metrics import get_logP, get_SA, calc_RingP

class selfies_env(object):
    '''This environment is build assuming selfies version 2.1.1
    To-do
    1) Register as an official gym environment
    '''
    def __init__(self, episode_length=72, max_selfies_len=72):
        # max selfies length was chosen because it is the maximum selfie length in the zinc250 dataset (selfies version 2.1.1)
        # Set maximum selfies length
        self.max_selfies_length = max_selfies_len

        # Set episode length
        self.episode_length = episode_length

        # Define action space 
        self.selfie_alphabet_set = sf.get_semantic_robust_alphabet()
        
        '''Remove sulphur and phosphorous based tokens from selfie_alphabet_set
        '''
        sulphur_tokens = []
        phosphorous_tokens = []

        for token in self.selfie_alphabet_set:
            if 'S' in token:
                sulphur_tokens.append(token) 
            if 'P' in token:
                phosphorous_tokens.append(token)

        for s_token in sulphur_tokens:
            self.selfie_alphabet_set.discard(s_token)
        for p_token in phosphorous_tokens:
            self.selfie_alphabet_set.discard(p_token)

        self.selfie_alphabet_set.add('[nop]')
        self.selfie_alphabet = sorted(list(self.selfie_alphabet_set))
        self.alphabet_to_idx = {s: i for i, s in enumerate(self.selfie_alphabet)}

        self.action_space = list(self.selfie_alphabet_set) # add a termination action ??

        # '''Adding phenyl group to alphabet
        # '''
        # smiles_benzene = "c1ccccc1"
        # selfies_benzene = sf.encoder(smiles_benzene)
        # self.action_space.append(selfies_benzene)

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

    def reward(self):
        molecule_smiles = sf.decoder(self.molecule_selfie)
        molecule = MolFromSmiles(molecule_smiles, sanitize=True)
        
        # normalize scores presumably according to zinc250 dataset. Code from https://github.com/aspuru-guzik-group/GA/blob/paper_results/4.1/random_selfies/random_baseline.py
        # reward = (( get_logP(molecule) - 2.4729421499641497 ) / 1.4157879815362406 ) - (( get_SA(molecule) - 3.0470797085649894 ) / 0.830643172314514 ) - (( calc_RingP(molecule) - 0.038131530820234766) / 0.2240274735210179 )
        
        reward = get_logP(molecule)
        return reward, molecule_smiles 
        
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
            reward, molecule_smiles = self.reward()
            info["smile"] = molecule_smiles
            info["episode"]["r"] = reward
            info["episode"]["l"] = self.t 
        else:
            reward, _ = self.reward()
        
        return self.onehot_selfies(self.molecule_selfie), reward, done, info

def get_best_step_solution(env, episode_len=1):
    score = -100 * np.ones(episode_len, len(env.action_space))

    for e_len in range(episode_len):
        molecule_selfie = ''
        for action_sfs in env.action_space:
            new_molecule_selfie = molecule_selfie + action_sfs

if __name__ == '__main__':
    np.random.seed(1)
    env = selfies_env()
   
    # logps = []
    # for sfs in env.action_space:
    #     smi = sf.decoder(sfs)
    #     mol = MolFromSmiles(smi)
    #     # img=Draw.MolsToGridImage([mol])
    #     logps.append(get_logP(mol))
    
    # best_action_id = np.argmax(logps)
    # best_action = env.action_space[best_action_id]

    # print(logps)
    # exit()
    # print(best_action)
    # print(logps[best_action_id])

    for i in range(1):
        state = env.reset()
        done = False
        while not done:
            action = np.random.randint(env.action_space_length)
            state, reward, done, info = env.step(action)