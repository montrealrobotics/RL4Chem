import group_selfies as group_sf
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from collections import OrderedDict

import numpy as np
from copy import deepcopy
import selfies as sf
import random

from dockstring import load_target

def prRed(prt): print("\033[91m {}\033[00m".format(prt), flush=True)
def prCyan(prt): print("\033[96m {}\033[00m".format(prt), flush=True)
def prLightGray(prt): print("\033[97m {}\033[00m".format(prt), flush=True)


class GroupSelfies_Molecule_Env:
    
    def __init__(self, grammar=None, action_size=100, dockstring_target='DRD2', episode_length=10):

        self.reset(grammar=grammar, action_size=action_size, dockstring_target=dockstring_target,
                   episode_length=episode_length)


    def reset(self, grammar=None, action_size=100, dockstring_target='DRD2', episode_length=10):

        # Create Molecular Action Space
        self.create_molecular_action_space(grammar=grammar, action_size=action_size)

        # Initialize Dockstring Target
        self.initialize_dockstring(target=dockstring_target)
        
        # Set Episode Length
        self.episode_length = episode_length
        
        # Initialize Molecule
        self.molecule = None

        # Initialize Step
        self.num_step = 0


    def create_molecular_action_space(self, grammar=None, action_size=100):

        self.grammar = grammar if grammar is not None else group_sf.group_grammar.common_r_group_replacements_grammar()

        smiles_strs = []
        selfies_strs = []
        tokens = []
        names = []
        for name, group in self.grammar.vocab.items():
            smiles = Chem.MolToSmiles(group.mol_without_attachment_points())
            selfies = sf.encoder(smiles)
            token = f"[:0{group.name}]"
            name = group.name

            smiles_strs.append(smiles)
            selfies_strs.append(selfies)
            tokens.append(token)
            names.append(name)

        self.action_space = random.sample(selfies_strs, action_size)


    def initialize_dockstring(self, target='DRD2'):

        self.dockstring_target = load_target(target)


    def evaluate_dockstring(self, ligand_smiles):

        score, aux = self.dockstring_target.dock(ligand_smiles)

        return score, aux

    def get_state(self):

        return self.grammar.decoder(self.molecule)
    
    def step(self, action, calc_dockstring=False):
        
        # Assume one-hot action - take out relevant string
        group_sf_index = torch.max(action)
        group_sf_token = self.action_space[group_sf_index]

        # Build the molecule - add fragment
        if self.molecule is None:
            self.molecule = group_sf_token
        else:
            mol_string = deepcopy(self.molecule)
            self.molecule += mol_string

        # Set Next State
        next_state = self.molecule

        # Check Step Number
        if self.num_step < self.episode_length and calc_dockstring is False:
            self.num_step += 1
            reward = 0; done = False; info = {}
            return next_state, reward, done, info
        else:
            mol_smiles = self.grammar.decoder(next_state)
            docking_info = self.evaluate_dockstring(ligand_smiles=mol_smiles)
            reward = docking_info[0]
            done = True
            info = {'docking_smiles': mol_smiles, 'docking_group_selfie': next_state,
                    'docking_score': reward, 'docking_affinities': docking_info[1]}
            self.reset()
            return next_state, reward, done, info


if __name__ == "__main__":

    test_env = GroupSelfies_Molecule_Env()

    import pdb; pdb.set_trace()