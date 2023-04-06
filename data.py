import os
import re
import torch
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from tdc.generation import MolGen
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import MolFromSmiles as smi2mol

class DockstringDataset:
    def __init__(self, vocab, add_bos, add_eos, device, data_file='data/docking-dataset.tsv', split_file='data/dockstring_split.tsv', train=True):
        dockstring_df = pd.read_csv(data_file, sep="\t")
        dockstring_splits = pd.read_csv(split_file, sep="\t")        

        if train:
            self.data = dockstring_df[dockstring_splits["split"] == "train"].smiles.tolist()
        else:
            self.data = dockstring_df[dockstring_splits["split"] == "test"].smiles.tolist()     

        self.vocab = vocab
        self.encoded_data = [vocab.encode(s, add_bos, add_eos) for s in self.data]
        self.len = [len(s) for s in self.encoded_data]
        self.max_len = np.max(self.len)
        self.device = device        
    
    def __len__(self):
        """
        Computes a number of objects in the dataset
        """
        return len(self.data)
   
    def __getitem__(self, index):
        encoded_tensor = torch.tensor(self.encoded_data[index], dtype=torch.long)
        return encoded_tensor[:-1], encoded_tensor[1:]

    def get_collate_fn(self, model_name):
        if model_name == 'char_rnn':
            def collate_fn(batch):
                x, next_x = list(zip(*batch))
                lens = [len(s) for s in x]               
                x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.vocab.pad).to(self.device)
                next_x = torch.nn.utils.rnn.pad_sequence(next_x, batch_first=True, padding_value=self.vocab.pad).to(self.device)
                return x, next_x, lens
        else:
            raise NotImplementedError
        
        return collate_fn
    
class smiles_vocabulary(object):
    def __init__(self, vocab_path='data/docstring_smiles_vocab.txt'):
        
        self.alphabet = set()
        with open(vocab_path, 'r') as f:
            chars = f.read().split()
        for char in chars:
            self.alphabet.add(char)
        
        self.special_tokens = ['EOS', 'BOS', 'PAD', 'UNK']

        self.alphabet_list = list(self.alphabet)
        self.alphabet_list.sort()
        self.alphabet_list = self.alphabet_list + self.special_tokens
        self.alphabet_length = len(self.alphabet_list)

        self.alphabet_to_idx = {s: i for i, s in enumerate(self.alphabet_list)}
        self.idx_to_alphabet = {s: i for i, s in self.alphabet_to_idx.items()}
    
    def tokenize(self, smiles, add_bos=False, add_eos=False):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        if add_bos:
            tokenized.insert(0, "BOS")
        if add_eos:
            tokenized.append('EOS')
        return tokenized
    
    def encode(self, smiles, add_bos=False, add_eos=False):
        """Takes a list of SMILES and encodes to array of indices"""
        char_list = self.tokenize(smiles, add_bos, add_eos)
        encoded_smiles = np.zeros(len(char_list), dtype=np.uint8)
        for i, char in enumerate(char_list):
            encoded_smiles[i] = self.alphabet_to_idx[char]
        return encoded_smiles

    def decode(self, encoded_smiles, rem_bos=True, rem_eos=True):
        """Takes an list of indices and returns the corresponding SMILES"""
        if rem_bos and encoded_smiles[0] == self.bos:
            encoded_smiles = encoded_smiles[1:]
        if rem_eos and encoded_smiles[-1] == self.eos:
            encoded_smiles = encoded_smiles[:-1]
            
        chars = []
        for i in encoded_smiles:
            chars.append(self.idx_to_alphabet[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def decode_padded(self, encoded_smiles, rem_bos=True):
        """Takes a padded array of indices and returns the corresponding SMILES"""
        if rem_bos and encoded_smiles[0] == self.bos:
            encoded_smiles = encoded_smiles[1:]
        
        chars = []
        for i in encoded_smiles:
            if i == self.eos: break
            chars.append(self.idx_to_alphabet[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles
    
    def __len__(self):
        return len(self.alphabet_to_idx)
    
    @property
    def bos(self):
        return self.alphabet_to_idx['BOS']
    
    @property
    def eos(self):
        return self.alphabet_to_idx['EOS']
    
    @property
    def pad(self):
        return self.alphabet_to_idx['PAD']
    
    @property
    def unk(self):
        return self.alphabet_to_idx['UNK']
    
def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string