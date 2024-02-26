"""Helper functions for computing the drug like properties for the 2000-2022 FDA Drug Set"""

import numpy as np
import math

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

def CalcNumElectronegative(mol):
    """
    count the number of electronegative atoms in a molecule
    """
    
    e_neg_atoms = ['F', 'O', 'Cl', 'N', 'Br', 'I']  # define e-neg atoms
    
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    total_e_neg = [atom for atom in atoms if atom in e_neg_atoms]
    
    return len(total_e_neg)


def GetSingleConfEnergy(mol):
    """
    get MMFF94-based energy for single random conformer
    note: RDKit struggles with certain atoms, primarily metals and Boron. Returns '-' to identify such cases.
    """
    
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=-1)
        
        return ff.CalcEnergy()
    
    except (AttributeError, ValueError) as e:
        
        return '-'


def GetMaxShapeDiff(mols):
    """
    Function to return the maximal shape difference and IDs of the two conformers
    with minimum shape similarity. (A score of 1 means identical, 0 is different, so lowest score = biggest shape difference)
    
    The idea here is that higher shape difference between conformers = more flexibility.

    Schreyer, A. M. and Blundell, T. USRCAT: real-time ultrafast shape recognition with pharmacophoric constraints, J. Chemoinf. 4 (2012)
    """
    
    # generate descriptors
    usrcat = [rdMolDescriptors.GetUSRCAT(mol) for mol in mols]
    
    # get scores between all conformer pairs
    scores = np.zeros((len(usrcat), len(usrcat)))
    for i, des1 in enumerate(usrcat):
        for j, des2 in enumerate(usrcat):
            scores[i][j] = rdMolDescriptors.GetUSRScore(des1, des2)

    return [round(np.amin(scores), 3), np.argwhere(scores == np.amin(scores))[0]]


def GetIsMacrocycle(mol):
    """
    Check if molecule is macrocyclic
    
    Yudin, A. K. Macrocycles: lessons from the distant past, recent developments,
    and future directions. Chem. Sci. 6, 30–49 (2015).
    """
    macro = Chem.MolFromSmarts("[r{12-}]") 
    if mol.HasSubstructMatch(macro):
        is_macro = 1
    else: 
        is_macro = 0
    return is_macro


def CalcBoltzAvg(e_list, prop_list):
    """
    get Boltzman Average for property
    """
    
    expon = [math.exp(-e / 0.593) for e in e_list] # boltzmann constant at room temp, units kcal⋅mol−1

    num = [prop_list[i] * expon[i] for i in range(len(prop_list))]
    
    boltz_avg = sum(num) / sum(expon)
    
    return boltz_avg

