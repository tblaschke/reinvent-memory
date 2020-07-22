# coding=utf-8

from typing import List

import numpy as np
from rdkit import Chem


class no_sulphur(object):
    """Scores structures based on not containing sulphur."""

    def __init__(self):
        pass

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        has_sulphur = [16 not in [atom.GetAtomicNum() for atom in mol.GetAtoms()] for mol in valid_mols]
        sulphur_score = [1 if ele else 0 for ele in has_sulphur]
        score = np.full(len(smiles), 0, dtype=np.float32)
        for idx, value in zip(valid_idxs, sulphur_score):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}
