import numpy as np
import selfies as sf
from torch.utils.data import Dataset

class Vocabulary(object):
    """A class for handling encoding/decoding from SELFIES to an array of indices"""
    def __init__(self, cfg):
        if cfg.init_from_file: 
            self.selfie_alphabet_set = self.init_from_file(cfg.init_from_file)
        else:
            self.selfie_alphabet_set = sf.get_semantic_robust_alphabet()
        
        self.selfie_alphabet_set.add('[nop]')
        self.selfie_alphabet = sorted(list(self.selfie_alphabet_set))
        self.alphabet_to_idx = {s: i for i, s in enumerate(self.selfie_alphabet)}

        if cfg.selfies_enc_type == 'one_hot':
            self.observation_shape = (self.max_selfies_length * len(self),)
            self.enc_selifes_fn = self.onehot_selfies
        elif cfg.selfies_enc_type == 'label':
            self.observation_shape = (self.max_selfies_length,)
            self.enc_selifes_fn = self.label_selfies

    def onehot_selfies(self, molecule_selfie):
        return np.array(sf.selfies_to_encoding(molecule_selfie, self.alphabet_to_idx, self.max_selfies_length, enc_type='one_hot'), dtype=np.uint8).flatten()

    def label_selfies(self, molecule_selfie):
        return np.array(sf.selfies_to_encoding(molecule_selfie, self.alphabet_to_idx, self.max_selfies_length, enc_type='label'), dtype=np.uint8).flatten()
    
    def init_from_file(self, file):
        raise NotImplementedError
        
    def __len__(self):
        return len(self.selfie_alphabet)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)