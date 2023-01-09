import numpy as np

import selfies as sf
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from metrics.metrics import get_logP, get_SA, calc_RingP

def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    
def get_random_mol(alphabet):
    '''Random walk in an MDP using elements of alphabet as actions and selfies formal grammar as transition dynamics.

    Parameters:
    alphabet (List[strings]) : List of selfie strings to be used as actions (building blocks) for the MDP

    Returns:
    mol (rdkit.Chem.rdchem.Mol) : Final molecule found from the random walk
    smiles_canon (string) : Canonicalized smile representation of the final molecule
    '''
    valid = False
    iter = 0
    while valid != True:
        selfie_str=''.join(np.random.choice(alphabet, 10))

        decoded_smile_str = sf.decoder(selfie_str)
        if decoded_smile_str != -1:
            valid = True
            
        # check if smile string is recognized by rdkit
        mol, smiles_canon, done = sanitize_smiles(decoded_smile_str)
        if mol == None or smiles_canon == '' or len(smiles_canon) > 81: 
            valid = False
            continue
        
        iter += 1
        if iter >= 100:
            raise Exception('Did not find a valid molecule after 100 trajectories')

    return mol, smiles_canon

if __name__ == '__main__':

    num_runs = 10
    max_scores = []

    '''Adding phenyl group to alphabet
    '''
    selfie_alphabet_set = sf.get_semantic_robust_alphabet()
    benzene_sf = '[C][=C][C][=C][C][=C][Ring1][=Branch1]'
    selfie_alphabet_set.add(benzene_sf)
    selfie_alphabets = list(selfie_alphabet_set)


    '''This was the alphabet used for 0.1.1
    # selfie_alphabets = ['[Branch1_1]', '[Branch1_2]','[Branch1_3]', '[epsilon]', '[Ring1]', '[Ring2]', '[Branch2_1]', '[Branch2_2]', '[Branch2_3]', '[F]', '[O]', '[=O]', '[N]', '[=N]', '[#N]', '[C]', '[=C]', '[#C]', '[S]', '[=S]', '[C][=C][C][=C][C][=C][Ring1][Branch1_1]']
    '''

    for i in range(num_runs):
        random_mols = []
        sm_random_mols = []

        for j in range(50000):
            
            random_mol, sm_random_mol = get_random_mol(selfie_alphabets)

            random_mols.append(random_mol)
            sm_random_mols.append(sm_random_mol)

            if len(sm_random_mols[j]) > 81:
                raise Exception('Length fail!')
            
        logP_scores   = []
        SA_scores    = []
        RingP_scores =  []

        print('Looking at i: ', i)
        for random_mol, sm_random_mol in zip(random_mols, sm_random_mols):

            logP_scores.append(get_logP(random_mol))
            SA_scores.append(get_SA(random_mol))
            RingP_scores.append(calc_RingP(random_mol))

            f = open("./results/results_{}.txt".format(i), "a+")
            f.write('smile: {} \n'.format(sm_random_mol))
            f.close()

        # normalize scores presumably according to zinc250 dataset. Code from https://github.com/aspuru-guzik-group/GA/blob/paper_results/4.1/random_selfies/random_baseline.py
        logP_norm  = [((x - 2.4729421499641497) / 1.4157879815362406) for x in logP_scores]
        SAS_norm   = [((x - 3.0470797085649894) / 0.830643172314514) for x in SA_scores]
        RingP_norm = [((x - 0.038131530820234766) / 0.2240274735210179) for x in RingP_scores]

        # Calculate J(m)
        J = []
        for i in range(len(logP_norm)):
            J.append(logP_norm[i] - SAS_norm[i] - RingP_norm[i])

        print('smile: ', sm_random_mols[J.index(max(J))], max(J))
        
        # Save result in text file
        f = open("results.txt", "a+")
        f.write('smile: {}, J: {} \n'.format(sm_random_mols[J.index(max(J))], max(J)))
        f.close()