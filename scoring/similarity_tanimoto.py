# coding=utf-8

from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

import utils


class tanimoto(object):
    """Scores based on the Tanimoto similarity to a query SMILES. Supports a similarity cutoff k."""

    def __init__(self, query_smi: utils.SMILES, k=1.0):
        self.k = k
        query_mol = Chem.MolFromSmiles(query_smi)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=False, useFeatures=True)

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = [AllChem.GetMorganFingerprint(mol, 2, useCounts=False, useFeatures=True) for mol in valid_mols]

        tanimoto = np.array([DataStructs.TanimotoSimilarity(self.query_fp, fp) for fp in fps])
        tanimoto = np.minimum(tanimoto, self.k) / self.k
        # tversky_similarity = np.square(tanimoto)
        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, tanimoto):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}
