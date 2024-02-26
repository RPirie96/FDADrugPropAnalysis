from rdkit.Chem import AllChem


from rdkit.Chem import AllChem, AddHs, RemoveHs, SDWriter, MolFromSmiles
import pandas as pd

def GenConformerEnsemble(f_in, f_out):
    """
    Function to generate multiple conformers of the Drug database and ouput the structures and their energy.
    Molecules RDKit cannot generate energies for (primarily ones with Boron) are marked with '-'
    
    Inputs:
    Name of excel file to read in: 'filename.xlsx'
    Name of SDF file to save outputs: 'filename.sdf'
    """
    
    # load data
    data = pd.read_excel(f_in)
    
    # extract 2D molecules
    mols = [MolFromSmiles(smile) for smile in list(data['SMILES'])]
    
    # set up parameters
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True  # includes recent improvements for small rings
    params.pruneRmsThresh = 0.1
    params.useRandomCoords = True
    params.randomSeed = 0xf00d
    params.maxAttempts = 1000
    
    
    # set up writer
    w = SDWriter(f_out)
    for i, mol in enumerate(mols):
        mol = AddHs(mol)  # add explicit hydrogens for better confs
        
        # get no. of conformers to generate based on no. rotatable bonds
        nb_rot_bonds = AllChem.CalcNumRotatableBonds(mol)
        if nb_rot_bonds <= 7:
            numconf = 50
        elif nb_rot_bonds <= 12:
            numconf = 200
        else:
            numconf = 300
        
        # get conformers
        cids = AllChem.EmbedMultipleConfs(mol, numconf, params)
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')
        
        #sort by energy
        res = []
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
            try:
                e = ff.CalcEnergy()
                res.append((cid, e))
            except:
                e = '-'
                res.append((cid, e))
        # sort by energy low -> high
        sorted_res = sorted(res, key=lambda x: (str(type(x)), x))  # sort by type then within type to handle placeholders
        
        mol = RemoveHs(mol)  # remove Hs
        
        #write confs
        for cid, e in sorted_res:
            mol.SetProp('MolNo', str(i))
            mol.SetProp('No Confs', str(numconf))
            mol.SetProp('No RBs', str(nb_rot_bonds))
            mol.SetProp('CID', str(cid))
            mol.SetProp('Energy', str(e))
            w.write(mol, confId=cid)
    w.close()