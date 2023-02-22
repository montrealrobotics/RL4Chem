import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import selfies as sf
from pathlib import Path
from models import CharRNN

def get_dataset(data_path='data/filtered_dockstring-dataset.tsv', splits_path='data/filtered_cluster_split.tsv', target='ESR2', string_rep='SMILES'):
    assert target in ['ESR2', 'F2', 'KIT', 'PARP1', 'PGR']
    dockstring_df = pd.read_csv(data_path)
    dockstring_splits = pd.read_csv(splits_path)

    bond_constraints = sf.get_semantic_constraints()
    bond_constraints['I'] = 5
    sf.set_semantic_constraints(bond_constraints)

    assert np.all(dockstring_splits.smiles == dockstring_df.smiles)
    assert sf.get_semantic_constraints()['I'] == 5

    df_train = dockstring_df[dockstring_splits["split"] == "train"].dropna(subset=[target])
    df_test = dockstring_df[dockstring_splits["split"] == "test"].dropna(subset=[target])
    
    y_train = df_train[target].values
    y_test = df_test[target].values

    y_train = np.minimum(y_train, 5.0)
    y_test = np.minimum(y_test, 5.0)

    if string_rep == 'SMILES':
        x_train = list(df_train['canon_smiles'])
        x_test = list(df_test['canon_smiles'])
    elif string_rep == 'SELFIES':
        assert sf == 1
        x_train = list(df_train['selfies'])
        x_test = list(df_test['selifes'])
    else:
        raise NotImplementedError

    return x_train, y_train, x_test, y_test

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

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

    def default_collate(self, batch):
        x, y = list(zip(*batch))
        lens = [len(s) for s in x]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.vocab.pad).to(self.device) 
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        return x, y, lens

vocab = smiles_vocabulary()
x_train, y_train, x_test, y_test = get_dataset()
train_dataset = StringDataset(vocab, x_train, y_train, device='cuda')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False, collate_fn=train_dataset.default_collate)

val_dataset = StringDataset(vocab, x_test, y_test, device='cuda')
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=train_dataset.default_collate)


model = CharRNN(vocab, device='cuda')
model.train()
def get_params():
    return (p for p in model.parameters() if p.requires_grad)
optimizer = optim.Adam(get_params(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

model.eval()
with torch.no_grad():
    for step, (x, y, lens) in enumerate(val_loader):
        preds = model(x, lens)
        loss = F.mse_loss(preds, y)
        print('val loss = ',  loss)

model.train()
for step, (x, y, lens) in enumerate(train_loader):
    preds = model(x, lens)
    loss = F.mse_loss(preds, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('train loss = ',  loss)

    

scheduler.step()
