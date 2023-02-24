import numpy as np
import selfies as sf

from rdkit import Chem
from pathlib import Path
from docking import DockingVina
from collections import defaultdict

class docking_env(object):
    '''This environment is build assuming selfies version 2.1.1
    To-do
    1) Register as an official gym environment
    '''
    def __init__(self, cfg):

        # Set maximum selfies length
        self.max_selfies_length = cfg.max_selfie_length

        # Set target property
        self.target = cfg.target
        self.docking_config = dict()
        if self.target == 'fa7':
            self.box_center = (10.131, 41.879, 32.097)
            self.box_size = (20.673, 20.198, 21.362)
            self.docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/fa7/receptor.pdbqt'
        elif self.target == 'parp1':
            self.box_center = (26.413, 11.282, 27.238)
            self.box_size = (18.521, 17.479, 19.995)
            self.docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/parp1/receptor.pdbqt'
        elif self.target == '5ht1b':
            self.box_center = (-26.602, 5.277, 17.898)
            self.box_size = (22.5, 22.5, 22.5)
            self.docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/5ht1b/receptor.pdbqt'
        else:
            raise NotImplementedError

        self.box_parameter = (self.box_center, self.box_size)
        self.docking_config['box_parameter'] = self.box_parameter
        self.docking_config['vina_program'] = cfg.vina_program
        self.docking_config['temp_dir'] = cfg.temp_dir
        self.docking_config['exhaustiveness'] = cfg.exhaustiveness
        self.docking_config['num_sub_proc'] = cfg.num_sub_proc
        self.docking_config['num_cpu_dock'] = cfg.num_cpu_dock
        self.docking_config['num_modes'] = cfg.num_modes
        self.docking_config['timeout_gen3d'] = cfg.timeout_gen3d
        self.docking_config['timeout_dock'] = cfg.timeout_dock

        self.predictor = DockingVina(self.docking_config)

        # Define alphabet 
        self.selfie_alphabet_set = sf.get_semantic_robust_alphabet()
        self.selfie_alphabet_set.add('[nop]')
        self.selfie_alphabet = sorted(list(self.selfie_alphabet_set))
        self.alphabet_length = len(self.selfie_alphabet_set)
        self.alphabet_to_idx = {s: i for i, s in enumerate(self.selfie_alphabet)}
        self.idx_to_alphabet = {s: i for i, s in self.alphabet_to_idx.items()}

        # Define action space 
        self.action_space = list(self.selfie_alphabet_set) # add a termination action ??
        self.action_space.sort()
        self.action_space_length = len(self.action_space)
        self.num_actions = self.action_space_length
        self.action_dtype = np.uint8
        assert self.alphabet_length == self.action_space_length    

        # Define observation space
        self.selfies_enc_type = cfg.selfies_enc_type
        if cfg.selfies_enc_type == 'one_hot':
            self.observation_shape = (self.max_selfies_length*self.alphabet_length,)
            self.enc_selifes_fn = self.onehot_selfies
            self.observation_dtype = np.uint8 
        elif cfg.selfies_enc_type == 'label':
            self.observation_shape = (self.max_selfies_length,)
            self.enc_selifes_fn = self.label_selfies
            self.observation_dtype = np.uint8 
        else:
            raise NotImplementedError

        # Initialize selfie string as benzene
        smiles_benzene = "c1ccccc1"
        self.init_molecule_selfie = sf.encoder(smiles_benzene)
        self.init_molecule_selfie_len = sf.len_selfies(self.init_molecule_selfie)

        # Set episode length
        self.episode_length = cfg.max_selfie_length - self.init_molecule_selfie_len

        # Intitialising smiles batch for parallel evaluation
        self.smiles_batch = []
        self.selfies_batch = []
        self.len_selfies_batch = []

        # Initialize Step
        self.t = 0

    def onehot_selfies(self, molecule_selfie):
        return np.array(sf.selfies_to_encoding(molecule_selfie, self.alphabet_to_idx, self.max_selfies_length, enc_type='one_hot'), dtype=self.observation_dtype).flatten()

    def label_selfies(self, molecule_selfie):
        return np.array(sf.selfies_to_encoding(molecule_selfie, self.alphabet_to_idx, self.max_selfies_length, enc_type='label'), dtype=self.observation_dtype).flatten()
    
    def reset(self):
        # Initialize selfie string
        self.molecule_selfie = self.init_molecule_selfie

        # Initialize Step
        self.t = 0

        return self.enc_selifes_fn(self.molecule_selfie)
    
    def step(self, action):
        assert self.t <= self.episode_length, 'episode has exceeded predefined limit, use env.reset()'
        assert action >=0 and action < self.action_space_length
        info = defaultdict(dict)

        action_selfie = self.action_space[action]
        self.molecule_selfie = self.molecule_selfie + action_selfie

        self.t += 1
        done = False
        if self.t >= self.episode_length:
            done = True
                    
        if done:
            molecule_smiles = sf.decoder(self.molecule_selfie)
            # pretty_selfies = sf.encoder(molecule_smiles)
            # self.selfies_batch.append(pretty_selfies)
            # self.len_selfies_batch.append(sf.len_selfies(pretty_selfies))
            info["episode"]["l"] = self.t
            info["episode"]["smiles"] = molecule_smiles
            reward = -1000
        else:
            reward = 0
        return self.enc_selifes_fn(self.molecule_selfie), reward, done, info
    
    def _add_smiles_to_batch(self, molecule_smiles):
        self.smiles_batch.append(molecule_smiles)

    def _reset_store_batch(self):
        # Intitialising smiles batch for parallel evaluation
        self.smiles_batch = []
        self.selfies_batch = []
        self.len_selfies_batch = []

    def get_reward_batch(self):
        info = defaultdict(dict)
        docking_scores = self.predictor.predict(self.smiles_batch)
        reward_batch = np.clip(-np.array(docking_scores), a_min=0.0, a_max=None)
        info['smiles'] = self.smiles_batch
        info['selfies'] = self.selfies_batch
        info['len_selfies'] = self.len_selfies_batch
        info['docking_scores'] = docking_scores
        self._reset_store_batch()
        return reward_batch, info

if __name__ == '__main__':
    from dataclasses import dataclass

    @dataclass
    class args():
        target= 'fa7'
        selfies_enc_type= 'label'
        max_selfie_length= 15
        vina_program= 'qvina2'
        temp_dir= 'tmp'
        exhaustiveness= 1
        num_sub_proc= 24
        num_cpu_dock= 1
        num_modes= 10
        timeout_gen3d= 30
        timeout_dock= 100

    env = docking_env(args)

'''
ENV stats
=======================================
Max len = 9
best_smiles = C1=CC=CC=C1N
best_reward = 5.0
=======================================
=======================================
Max len = 10
best_smiles = C1=CC=CC=C1[N+1]=N, C1=CC=CC=C1N=N, C1=CC=CC=C1[O+1]=N ...
best_reward = 5.6
=======================================
'''