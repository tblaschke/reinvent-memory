# coding=utf-8

import logging
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Pharm2D import Generate

from chem.pharmacophore import factory
from utils import Timeout, utils


class pharmacophoric(object):
    """Scores based on pharmacophoric similarity to query ligand."""

    def __init__(self, query_smi: utils.SMILES):

        self.sigFactory = factory
        query = Chem.MolFromSmiles(query_smi)
        self.query_fp = Generate.Gen2DFingerprint(query, self.sigFactory)

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        similarity = [self.tanimoto(mol) for mol in valid_mols]
        # similarity = np.square(similarity)

        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, similarity):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}

    def tanimoto(self, mol):
        try:
            with Timeout(seconds=1):
                fp = Generate.Gen2DFingerprint(mol, self.sigFactory)
            return DataStructs.TanimotoSimilarity(fp, self.query_fp)

        except TimeoutError:
            logging.debug("SMILES Pharmacophore timeout: ", Chem.MolToSmiles(mol, isomericSmiles=False))
            return 0
