# coding=utf-8

from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class clogp(object):
    """ Optimize strutures to have a predicted clogP within a particular range"""

    def __init__(self, range: str):
        try:
            numbers = list(map(float, range.split("-")))
        except:
            raise ValueError("clogp needs to be a valid float number or a valid range like: 2.0-3.0")
        if len(numbers) == 1:
            self.clogp = (numbers,numbers)
        else:
            self.clogp = (numbers[0], numbers[1])

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        
        clogp = [ self.calcLogP(mol) for mol in valid_mols ]
        logp_score = [self.score_clogp(ele) for ele in clogp]
        score = np.full(len(smiles), 0, dtype=np.float32)
        for idx, value in zip(valid_idxs, logp_score):
            score[idx] = value
        #return {"total_score": np.array(score, dtype=np.float32), "clogp": clogp}
        return {"total_score": np.array(score, dtype=np.float32)}
        
    def calcLogP(self, mol):
        try:
            return rdMolDescriptors.CalcCrippenDescriptors(mol)[0] 
        except:
            return None 
        
    def score_clogp(self, clogp):
        if clogp is None:
            return 0
        else:
            if clogp < self.clogp[0]:
                min_distance = abs(self.clogp[0] - clogp)
            else: #greater equal than lower bound
                if clogp <= self.clogp[1]: #right in the boundary
                    return 1
                else: 
                    distance_to_lower_bound = abs(self.clogp[0] - clogp)
                    distance_to_upper_bound = abs(self.clogp[1] - clogp)
                    min_distance = min(distance_to_lower_bound, distance_to_upper_bound)
            
            #transfor distance to score between 0 and 1
            return 1 - np.tanh(min_distance) 