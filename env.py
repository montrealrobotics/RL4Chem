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

        # Set episode length
        self.episode_length = cfg.max_selfie_length

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

        # Define observation space
        assert self.alphabet_length == self.action_space_length    

        self.selfies_enc_type = cfg.selfies_enc_type
        if cfg.selfies_enc_type == 'one_hot':
            self.observation_shape = (self.max_selfies_length*self.alphabet_length,)
            self.enc_selifes_fn = self.onehot_selfies
            self.observation_dtype = np.uint8 
        elif cfg.selfies_enc_type == 'label':
            self.observation_shape = (self.max_selfies_length,)
            self.enc_selifes_fn = self.label_selfies
        else:
            raise NotImplementedError

        # Initialize selfie string as benzene
        smiles_benzene = "c1ccccc1"
        self.init_molecule_selfie = sf.encoder(smiles_benzene)

        # Intitialising smiles batch for parallel evaluation
        self.smiles_batch = []

        # Initialize Step
        self.t = 0

    def onehot_selfies(self, molecule_selfie):
        return np.array(sf.selfies_to_encoding(molecule_selfie, self.alphabet_to_idx, self.max_selfies_length, enc_type='one_hot'), dtype=np.uint8).flatten()

    def label_selfies(self, molecule_selfie):
        return np.array(sf.selfies_to_encoding(molecule_selfie, self.alphabet_to_idx, self.max_selfies_length, enc_type='label'), dtype=np.uint8).flatten()
    
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
            self.smiles_batch.append(molecule_smiles)
            reward = None
            info["episode"]["l"] = self.t
            info['smiles'] = molecule_smiles
        else:
            reward = 0
        return self.enc_selifes_fn(self.molecule_selfie), reward, done, info
    
    def reset_smiles_batch(self):
        # Intitialising smiles batch for parallel evaluation
        self.smiles_batch = []

    def get_reward_batch(self):
        info = defaultdict(dict)
        docking_scores = self.predictor.predict(self.smiles_batch)
        reward_batch = np.clip(-np.array(docking_scores), a_min=0.0, a_max=None)
        info['smiles'] = self.smiles_batch
        info['docking_scores'] = docking_scores
        return reward_batch, info

if __name__ == '__main__':
    import time
    start = time.time()
    target = 'fa7'
    docking_config = dict()

    if target == 'fa7':
        box_center = (10.131, 41.879, 32.097)
        box_size = (20.673, 20.198, 21.362)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/fa7/receptor.pdbqt'
    elif target == 'parp1':
        box_center = (26.413, 11.282, 27.238)
        box_size = (18.521, 17.479, 19.995)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/parp1/receptor.pdbqt'
    elif target == '5ht1b':
        box_center = (-26.602, 5.277, 17.898)
        box_size = (22.5, 22.5, 22.5)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/5ht1b/receptor.pdbqt'

    box_parameter = (box_center, box_size)
    docking_config['vina_program'] = 'qvina2'
    docking_config['box_parameter'] = box_parameter
    docking_config['temp_dir'] = 'tmp'
    docking_config['exhaustiveness'] = 1
    docking_config['num_sub_proc'] = 12
    docking_config['num_cpu_dock'] = 12
    docking_config['num_modes'] = 10 
    docking_config['timeout_gen3d'] = 30
    docking_config['timeout_dock'] = 100 

    predictor = DockingVina(docking_config)

    selfie_alphabet_set = sf.get_semantic_robust_alphabet()
    smiles_benzene = "c1ccccc1"
    selfies_benzene = sf.encoder(smiles_benzene)
    selfie_alphabet_set.add(selfies_benzene)
    selfie_alphabets = list(selfie_alphabet_set)

    smiles_list = []
    for i in range(10):
        selfie_str=''.join(np.random.choice(selfie_alphabets, 40))
        decoded_smile_str = sf.decoder(selfie_str)
        smiles_list.append(decoded_smile_str)
    
    a_list = predictor.predict(smiles_list)
    print(a_list)
    print(time.time()-start)