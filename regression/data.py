import re
import torch
import numpy as np
import pandas as pd
import selfies as sf

from torch.utils.data import DataLoader

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

class selfies_vocabulary(object):
    def __init__(self, vocab_path='data/dockstring_selfies_vocabulary.txt', robust_alphabet=False):
    
        if robust_alphabet:
            self.alphabet = sf.get_semantic_robust_alphabet()
        else:
            self.alphabet = set()
            with open(vocab_path, 'r') as f:
                chars = f.read().split()
            for char in chars:
                self.alphabet.add(char)

        self.special_tokens = ['BOS', 'EOS', 'PAD', 'UNK']

        self.alphabet_list = list(self.alphabet)
        self.alphabet_list.sort()
        self.alphabet_list = self.alphabet_list + self.special_tokens
        self.alphabet_length = len(self.alphabet_list)

        self.alphabet_to_idx = {s: i for i, s in enumerate(self.alphabet_list)}
        self.idx_to_alphabet = {s: i for i, s in self.alphabet_to_idx.items()}

    def tokenize(self, selfies, add_bos=False, add_eos=False):
        """Takes a SELFIES and return a list of characters/tokens"""
        tokenized = list(sf.split_selfies(selfies))
        if add_bos:
            tokenized.insert(0, "BOS")
        if add_eos:
            tokenized.append('EOS')
        return tokenized

    def encode(self, selfies, add_bos=False, add_eos=False):
        """Takes a list of SELFIES and encodes to array of indices"""
        char_list = self.tokenize(selfies, add_bos, add_eos)
        encoded_selfies = np.zeros(len(char_list), dtype=np.uint8)
        for i, char in enumerate(char_list):
            encoded_selfies[i] = self.alphabet_to_idx[char]
        return encoded_selfies

    def decode(self, encoded_seflies, rem_bos=True, rem_eos=True):
        """Takes an array of indices and returns the corresponding SELFIES"""
        if rem_bos and encoded_seflies[0] == self.bos:
            encoded_seflies = encoded_seflies[1:]
        if rem_eos and encoded_seflies[-1] == self.eos:
            encoded_seflies = encoded_seflies[:-1]
            
        chars = []
        for i in encoded_seflies:
            chars.append(self.idx_to_alphabet[i])
        selfies = "".join(chars)
        return selfies

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
    
class smiles_vocabulary(object):
    def __init__(self, vocab_path='data/dockstring_smiles_vocabulary.txt'):
        
        self.alphabet = set()
        with open(vocab_path, 'r') as f:
            chars = f.read().split()
        for char in chars:
            self.alphabet.add(char)
        
        self.special_tokens = ['BOS', 'EOS', 'PAD', 'UNK']

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
        """Takes an array of indices and returns the corresponding SMILES"""
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

class StringDataset:
    def __init__(self, vocab, data, target, device, add_bos=False, add_eos=False):
        """
        Arguments:
            vocab: CharVocab instance for tokenization
            data (list): SMILES/SELFIES strings for the datasety
            target (arra): Array of target values
            target (list): 
        """
        self.data = data
        self.target = target
        self.vocab = vocab
        self.device = device
        self.encoded_data = [vocab.encode(s, add_bos, add_eos) for s in data]

    def __len__(self):
        """
        Computes a number of objects in the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.encoded_data[index], dtype=torch.long), self.target[index]

    def get_collate_fn(self, model_name):
        if model_name == 'char_rnn':
            def collate_fn(batch):
                x, y = list(zip(*batch))
                lens = [len(s) for s in x]
                x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.vocab.pad).to(self.device) 
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
                return x, y, lens
            return collate_fn
        else:
            raise NotImplementedError
    
def get_data(cfg):
    data_path=cfg.data_path
    splits_path=cfg.splits_path
    target=cfg.target
    string_rep=cfg.string_rep
    batch_size=cfg.batch_size

    assert target in ['ESR2', 'F2', 'KIT', 'PARP1', 'PGR']
    dockstring_df = pd.read_csv(data_path)
    dockstring_splits = pd.read_csv(splits_path)

    assert np.all(dockstring_splits.smiles == dockstring_df.smiles)
    
    df_train = dockstring_df[dockstring_splits["split"] == "train"].dropna(subset=[target])
    df_test = dockstring_df[dockstring_splits["split"] == "test"].dropna(subset=[target])
    
    y_train = df_train[target].values
    y_test = df_test[target].values

    y_train = np.minimum(y_train, 5.0)
    y_test = np.minimum(y_test, 5.0)

    if string_rep == 'smiles':
        x_train = list(df_train['canon_smiles'])
        x_test = list(df_test['canon_smiles'])
        vocab = smiles_vocabulary()

    elif string_rep == 'selfies':
        assert sf.__version__ == '2.1.0'
        bond_constraints = sf.get_semantic_constraints()
        bond_constraints['I'] = 5
        sf.set_semantic_constraints(bond_constraints)
        assert sf.get_semantic_constraints()['I'] == 5

        x_train = list(df_train['selfies_'+sf.__version__])
        x_test = list(df_test['selfies_'+sf.__version__])
        vocab = selfies_vocabulary()
    else:
        raise NotImplementedError

    train_dataset = StringDataset(vocab, x_train, y_train, device='cuda')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=train_dataset.get_collate_fn(cfg.model_name))
    val_dataset = StringDataset(vocab, x_test, y_test, device='cuda')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=train_dataset.get_collate_fn(cfg.model_name))

    return train_loader, val_loader, vocab