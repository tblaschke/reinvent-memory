# coding=utf-8

from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

import utils


class tanimoto_file(object):
    """Scores based on the Tanimoto similarity all SMILES in a file."""

    def __init__(self, file_path: utils.FilePath):
        mols = []
        with open(file_path, 'r') as f:
            for line in f:
                smile = line.strip().split()[0]
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    mols.append(mol)
        self.ref_fps = [AllChem.GetMorganFingerprint(mol,
                                                     3, useCounts=True, useFeatures=False) for mol in mols]

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = [AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=False) for mol in valid_mols]

        tanimoto = np.array([np.max(DataStructs.BulkTanimotoSimilarity(fp, self.ref_fps)) for fp in fps])
        tanimoto = np.maximum((1 - 2 * np.absolute(0.5 - tanimoto)), 0)

        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, tanimoto):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}
