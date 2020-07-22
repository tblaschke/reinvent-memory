# coding=utf-8

import logging

from joblib import Parallel, delayed
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import SaltRemover
from rdkit.Chem import rdmolops

rdBase.DisableLog('rdApp.error')


def _initialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
        )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


_reactions = _initialiseNeutralisationReactions()


def _neutraliseCharges(mol, reactions=None):
    global _reactions
    if reactions is None:
        reactions = _reactions
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return mol, True
    else:
        return mol, False


def _getlargestFragment(mol):
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    maxmol = None
    for mol in frags:
        if mol is None:
            continue
        if maxmol is None:
            maxmol = mol
        if maxmol.GetNumHeavyAtoms() < mol.GetNumHeavyAtoms():
            maxmol = mol
    return maxmol


_saltremover = SaltRemover.SaltRemover()


def valid_size(mol, min_heavy_atoms, max_heavy_atoms, element_list, remove_long_side_chains):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol:
        correct_size = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
        if not correct_size:
            return

        valid_elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if not valid_elements:
            return

        has_long_sidechains = False
        if remove_long_side_chains:
            # remove aliphatic side chains with at least 4 carbons not in a ring
            sma = '[CR0]-[CR0]-[CR0]-[CR0]'
            has_long_sidechains = mol.HasSubstructMatch(Chem.MolFromSmarts(sma))

        return correct_size and valid_elements and not has_long_sidechains


def standardize_smiles(smiles, min_heavy_atoms=10, max_heavy_atoms=50, element_list=[6, 7, 8, 9, 16, 17, 35],
                       remove_long_side_chains=False, neutralise_charges=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = _getlargestFragment(mol)
    if mol:
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol:
        mol = _saltremover.StripMol(mol, dontRemoveEverything=True)
    if mol and neutralise_charges:
        mol, _ = _neutraliseCharges(mol)
    if mol:
        rdmolops.Cleanup(mol)
        rdmolops.SanitizeMol(mol)
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol and valid_size(mol, min_heavy_atoms, max_heavy_atoms, element_list, remove_long_side_chains):
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    return None


def standardize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = [line.strip().split(" ")[0] for line in f]
    return standardize_smiles_list(smiles_list)


def standardize_smiles_list(smiles_list):
    """Reads a SMILES list and returns a list of RDKIT SMILES"""
    smiles_list = Parallel(n_jobs=-1, verbose=0)(delayed(standardize_smiles)(line) for line in smiles_list)
    smiles_list = [smiles for smiles in set(smiles_list) if smiles is not None]
    logging.debug("{} unique SMILES retrieved".format(len(smiles_list)))
    return smiles_list
