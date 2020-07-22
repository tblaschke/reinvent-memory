# coding=utf-8

import logging
import pickle
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy.sparse
from sklearn import calibration
import joblib

import utils


class activity_hard(object):
    """Scores based on an ECFP6 classifier for activity (proba * decision)"""

    def __init__(self, clf_path: utils.FilePath):
        logging.debug(clf_path)
        self.clf_path = clf_path
        self.clf = joblib.load(clf_path)
        if isinstance(self.clf, calibration.CalibratedClassifierCV):
            for c in self.clf.calibrated_classifiers_:
                if c.base_estimator.kernel == "tanimoto":
                    c.base_estimator.kernel = tanimotokernel
                elif c.base_estimator.kernel == "minmax":
                    c.base_estimator.kernel = minmaxkernel
        else:
            if self.clf.kernel == "tanimoto":
                self.clf.kernel = tanimotokernel
            elif self.clf.kernel == "minmax":
                self.clf.kernel = minmaxkernel

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        score = np.full(len(smiles), 0, dtype=np.float32)

        fps = activity_hard.fingerprints_from_mols(valid_mols)
        activity_score = self.clf.predict_proba(fps)[:, 1] *  self.clf.predict(fps)

        for idx, value in zip(valid_idxs, activity_score):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}

    @classmethod
    def fingerprints_from_mols(cls, mols):
        fps = [AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=False) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.float64)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp

    def __reduce__(self):
        """
        :return: A tuple with the constructor and its arguments. Used to reinitialize the object for pickling
        """
        return activity_hard, (self.clf_path,)


def linearkernel(data_1, data_2):
    return np.dot(data_1, data_2.T)


def tanimotokernel(data_1, data_2):
    if isinstance(data_1, scipy.sparse.csr_matrix) and isinstance(data_2, scipy.sparse.csr_matrix):
        return _sparse_tanimotokernel(data_1, data_2)
    elif isinstance(data_1, scipy.sparse.csr_matrix) or isinstance(data_2, scipy.sparse.csr_matrix):
        # try to sparsify the input
        return _sparse_tanimotokernel(scipy.sparse.csr_matrix(data_1), scipy.sparse.csr_matrix(data_2))
    else:  # both are dense
        return _dense_tanimotokernel(data_1, data_2)
     
    
def _dense_tanimotokernel(data_1, data_2):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)
    as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    https://www.sciencedirect.com/science/article/pii/S0893608005001693
    http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Ralaivola2005Graph.pdf
    """

    norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
    norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
    prod = data_1.dot(data_2.T)

    divisor = (norm_1 + norm_2.T - prod) + np.finfo(data_1.dtype).eps
    return prod / divisor


def _sparse_tanimotokernel(data_1, data_2):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)
    as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    https://www.sciencedirect.com/science/article/pii/S0893608005001693
    http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Ralaivola2005Graph.pdf
    """

    norm_1 = np.array(data_1.power(2).sum(axis=1).reshape(data_1.shape[0], 1))
    norm_2 = np.array(data_2.power(2).sum(axis=1).reshape(data_2.shape[0], 1))
    prod = data_1.dot(data_2.T).A

    divisor = (norm_1 + norm_2.T - prod) + np.finfo(data_1.dtype).eps
    result = prod / divisor
    return result


def _minmaxkernel_numpy(data_1, data_2):
    """
    MinMax kernel
        K(x, y) = SUM_i min(x_i, y_i) / SUM_i max(x_i, y_i)
    bounded by [0,1] as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.483&rep=rep1&type=pdf
    """
    return np.stack([(np.minimum(data_1, data_2[cpd,:]).sum(axis=1) / np.maximum(data_1, data_2[cpd,:]).sum(axis=1))  for cpd in range(data_2.shape[0])],axis=1)


try: 
    from numba import njit, prange

    @njit(parallel=True, fastmath=True)
    def _minmaxkernel_numba(data_1, data_2):
        """
        MinMax kernel
            K(x, y) = SUM_i min(x_i, y_i) / SUM_i max(x_i, y_i)
        bounded by [0,1] as defined in:
        "Graph Kernels for Chemical Informatics"
        Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
        Neural Networks
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.483&rep=rep1&type=pdf
        """


        result = np.zeros((data_1.shape[0], data_2.shape[0]))

        for i in prange(data_1.shape[0]):
            for j in prange(data_2.shape[0]):
                result[i,j] = _minmax_two_fp(data_1[i], data_2[j])
        return result


    @njit(fastmath=True)
    def _minmax_two_fp(fp1, fp2):
        common = 0.
        maxnum = 0.
        for i in range(len(fp1)):
            min_ = fp1[i]
            max_ = fp2[i]

            if min_ > max_:
                min_ = fp2[i]
                max_ = fp1[i]

            common += min_
            maxnum += max_
        return common/maxnum

    minmaxkernel = _minmaxkernel_numba
    
except:
    
    logging.warning("Couldn't find numba. I suggest to install numba to compute the minmax kernel much much faster")
    
    minmaxkernel = _minmaxkernel_numpy

