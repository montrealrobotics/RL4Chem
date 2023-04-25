import re
import numpy as np
import selfies as sf

class selfies_vocabulary(object):
    def __init__(self, vocab_path):
        
        self.alphabet = set()
        with open(vocab_path, 'r') as f:
            chars = f.read().split()
        for char in chars:
            self.alphabet.add(char)
        
        self.special_tokens = ['[EOS]', '[BOS]', '[PAD]', '[UNK]']

        self.alphabet_list = list(self.alphabet)
        self.alphabet_list.sort()
        self.alphabet_list = self.alphabet_list + self.special_tokens
        self.alphabet_length = len(self.alphabet_list)

        self.alphabet_to_idx = {s: i for i, s in enumerate(self.alphabet_list)}
        self.idx_to_alphabet = {s: i for i, s in self.alphabet_to_idx.items()}

        self.action_list = self.alphabet_list[:-3]
        self.action_length = len(self.action_list)

        self.special_tokens_idx = [self.eos, self.bos, self.pad, self.unk]
    
    def tokenize(self, selfies, add_bos=False, add_eos=False):
        """Takes a SMILES and return a list of characters/tokens"""
        char_list = sf.split_selfies(selfies)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        for char in char_list:
            tokenized.append(char)
        if add_bos:
            tokenized.insert(0, "[BOS]")
        if add_eos:
            tokenized.append('[EOS]')
        return tokenized

    def encode(self, selfies, add_bos=False, add_eos=False):
        """Takes a list of SELFIES and encodes to array of indices"""
        char_list = self.tokenize(selfies, add_bos, add_eos)
        encoded_selfies = np.zeros(len(char_list), dtype=np.uint8)
        for i, char in enumerate(char_list):
            encoded_selfies[i] = self.alphabet_to_idx[char]
        return encoded_selfies

    def decode(self, encoded_selfies, rem_bos=True, rem_eos=True):
        """Takes an list of indices and returns the corresponding SELFIES"""
        if rem_bos and encoded_selfies[0] == self.bos:
            encoded_selfies = encoded_selfies[1:]
        if rem_eos and encoded_selfies[-1] == self.eos:
            encoded_selfies = encoded_selfies[:-1]
            
        chars = []
        for i in encoded_selfies:
            chars.append(self.idx_to_alphabet[i])
        selfies = "".join(chars)
        smiles = sf.decoder(selfies)
        return smiles
    
    def decode_padded(self, encoded_selfies, rem_bos=True):
        """Takes a padded array of indices which might contain special tokens and returns the corresponding SMILES"""
        if rem_bos and encoded_selfies[0] == self.bos:
            encoded_selfies = encoded_selfies[1:]
        
        chars = []
        for i in encoded_selfies:
            if i == self.eos: break

            if i not in self.special_tokens_idx: chars.append(self.idx_to_alphabet[i])
            
        selfies = "".join(chars)
        smiles = sf.decoder(selfies)
        return smiles

    def __len__(self):
        return len(self.alphabet_to_idx)
    
    @property
    def bos(self):
        return self.alphabet_to_idx['[BOS]']
    
    @property
    def eos(self):
        return self.alphabet_to_idx['[EOS]']
    
    @property
    def pad(self):
        return self.alphabet_to_idx['[PAD]']
    
    @property
    def unk(self):
        return self.alphabet_to_idx['[UNK]']
        
class smiles_vocabulary(object):
    def __init__(self, vocab_path):
        
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

        self.special_tokens_idx = [self.eos, self.bos, self.pad, self.unk]
        
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

            if i not in self.special_tokens_idx: chars.append(self.idx_to_alphabet[i])

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