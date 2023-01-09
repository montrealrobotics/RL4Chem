from rdkit.Chem import Descriptors
from metrics.rdkit_metric.sascorer import calculateScore

def get_SA(mol):
    '''Calculate synthetic accessibility (SA) score of a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object, for SA score calculation
    
    Returns:
    float : sac of molecule (mol)
    '''
    return calculateScore(mol)

def get_logP(mol):
    '''Calculate logP of a molecule.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object, for logP calculation
    
    Returns:
    float : logP of molecule (mol)
    '''
    return Descriptors.MolLogP(mol)

def calc_RingP(mol):
    '''Calculate ring penalty for a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object, for ring penalty calculation
    
    Returns:
    float : ring penalty of molecule (mol)
    '''
    cycle_list = mol.GetRingInfo().AtomRings() 
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length