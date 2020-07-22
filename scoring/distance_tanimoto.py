# coding=utf-8

from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

import utils


class tanimoto_distance(object):
    """Scores based on the tanimoto distance to a query SMILES."""

    def __init__(self, query_smi: utils.SMILES):
        query_mol = Chem.MolFromSmiles(query_smi)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=False, useFeatures=True)

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = [AllChem.GetMorganFingerprint(mol, 2, useCounts=False, useFeatures=True) for mol in valid_mols]

        tanimoto_dist = np.array([DataStructs.TanimotoSimilarity(self.query_fp, fp, returnDistance=True) for fp in fps])
        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, tanimoto_dist):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}
