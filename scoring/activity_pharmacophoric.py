# coding=utf-8

import pickle
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate

import utils
from chem.pharmacophore import factory


class activity_model_pharmacophoric(object):
    """Predicts activity with a pharmacophore fingerprint classifier.
       Also includes both desirable and undesirable fragments/SMARTS."""

    def __init__(self, clf_path: utils.FilePath, ecfp_clf_path: utils.FilePath):
        with open(clf_path, "rb") as f:
            self.clf = pickle.load(f)
        with open(ecfp_clf_path, "rb") as f:
            self.ecfp_clf = pickle.load(f)
        self.qed_alerts = [
            "*1[O,S,N]*1",
            "[S,C](=[O,S])[F,Br,Cl,I]",
            "[CX4][Cl,Br,I]",
            "[C,c]S(=O)(=O)O[C,c]",
            "[$([CH]),$(CC)]#CC(=O)[C,c]",
            "[$([CH]),$(CC)]#CC(=O)O[C,c]",
            "n[OH]",
            "[$([CH]),$(CC)]#CS(=O)(=O)[C,c]",
            "C=C(C=O)C=O",
            "n1c([F,Cl,Br,I])cccc1",
            "[CH1](=O)",
            "[O,o][O,o]",
            "[C;!R]=[N;!R]",
            "[N!R]=[N!R]",
            "[#6](=O)[#6](=O)",
            "[S,s][S,s]",
            "[N,n][NH2]",
            "C(=O)N[NH2]",
            "[C,c]=S",
            "[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]",
            "C1(=[O,N])C=CC(=[O,N])C=C1",
            "C1(=[O,N])C(=[O,N])C=CC=C1",
            "a21aa3a(aa1aaaa2)aaaa3",
            "a31a(a2a(aa1)aaaa2)aaaa3",
            "a1aa2a3a(a1)A=AA=A3=AA=A2",
            "c1cc([NH2])ccc1",
            "[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,\
            Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]",
            "I",
            "OS(=O)(=O)[O-]",
            "[N+](=O)[O-]",
            "C(=O)N[OH]",
            "C1NC(=O)NC(=O)1",
            "[SH]",
            "[S-]",
            "c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]",
            "c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]",
            "[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1",
            "[CR1]1[CR1][CR1]cc[CR1][CR1]1",
            "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1",
            "[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1",
            "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
            "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
            "C#C",
            "[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]",
            "[$([N+R]),$([n+R]),$([N+]=C)][O-]",
            "[C,c]=N[OH]",
            "[C,c]=NOC=O",
            "[C,c](=O)[CX4,CR0X3,O][C,c](=O)",
            "c1ccc2c(c1)ccc(=O)o2",
            "[O+,o+,S+,s+]",
            "N=C=O",
            "[NX3,NX4][F,Cl,Br,I]",
            "c1ccccc1OC(=O)[#6]",
            "[CR0]=[CR0][CR0]=[CR0]",
            "[C+,c+,C-,c-]",
            "N=[N+]=[N-]",
            "C12C(NC(N1)=O)CSC2",
            "c1c([OH])c([OH,NH2,NH])ccc1",
            "P",
            "[N,O,S]C#N",
            "C=C=O",
            "[SX2]O",
            "[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)",
            "O1CCCCC1OC2CCC3CCCCC3C2",
            "N=[CR0][N,n,O,S]",
            "[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]\
            ([Nv3X3,Nv4X4])[cR2][cR2]2",
            "C=[C!r]C#N",
            "[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1",
            "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1",
            "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])",
            "[OH]c1ccc([OH,NH2,NH])cc1",
            "c1ccccc1OC(=O)O",
            "[SX2H0][N]",
            "c12ccccc1(SC(S)=N2)",
            "c12ccccc1(SC(=S)N2)",
            "c1nnnn1C=O",
            "s1c(S)nnc1NC=O",
            "S1C=CSC1=S",
            "C(=O)Onnn",
            "OS(=O)(=O)C(F)(F)F",
            "N#CC[OH]",
            "N#CC(=O)",
            "S(=O)(=O)C#N",
            "N[CH2]C#N",
            "C1(=O)NCC1",
            "S(=O)(=O)[O-,OH]",
            "NC[F,Cl,Br,I]",
            "C=[C!r]O",
            "[NX2+0]=[O+0]",
            "[OR0,NR0][OR0,NR0]",
            "C(=O)O[C,H].C(=O)O[C,H].C(=O)O[C,H]",
            "[CX2R0][NX3R0]",
            "c1ccccc1[C;!R]=[C;!R]c2ccccc2",
            "[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]",
            "[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]",
            "[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]",
            "[*]=[N+]=[*]",
            "[SX3](=O)[O-,OH]",
            "N#N",
            "F.F.F.F",
            "[R0;D2][R0;D2][R0;D2][R0;D2]",
            "[cR,CR]~C(=O)NC(=O)~[cR,CR]",
            "C=!@CC=[O,S]",
            "[#6,#8,#16][C,c](=O)O[C,c]",
            "c[C;R0](=[O,S])[C,c]",
            "c[SX2][C;!R]",
            "C=C=C",
            "c1nc([F,Cl,Br,I,S])ncc1",
            "c1ncnc([F,Cl,Br,I,S])c1",
            "c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])",
            "[C,c]S(=O)(=O)c1ccc(cc1)F",
            "[15N,13C,18O,2H,34S]"]
        # These custom alerts are used to filter out very common kinase binding motifs
        self.custom_alerts = [
            '[NH2,NH1][a]1n[a][a][a][a]1',
            'NC(=N)',
            'Nc1ncnc2ccccc21',
            'Nc1ncnc(N)c1',
            'c1ccnc2[nH]ccc21']

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = activity_model_pharmacophoric.fingerprints_from_mols(valid_mols)
        # ecfp_fps = activity_model_pharmacophoric.ecfp_from_mols(valid_mols)
        activity_score = self.clf.predict_proba(fps)[:, 1]
        # ecfp_activity_score = self.ecfp_clf.predict_proba(ecfp_fps)[:, 1]
        subst = self.substructure(valid_mols, ['[NH2,NH1][a]1[a][a][a][a]1C(=O)[Nh]'])
        qed_alerts = self.substructure(valid_mols, self.qed_alerts)
        custom_alerts = self.substructure(valid_mols, self.custom_alerts)
        num_atoms = np.array([max(0, min(2.66 - mol.GetNumHeavyAtoms() / 15, 1)) for mol in valid_mols])

        goodness = 0.5 * (1 + subst) * (1 - custom_alerts) * (1 - qed_alerts) * num_atoms
        # goodness = 0.5 * (1 + subst) * (1 - custom_alerts) * num_atoms

        activity_score = activity_score * goodness
        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, activity_score):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}

    @classmethod
    def fingerprints_from_mols(cls, mols):
        fps = [Generate.Gen2DFingerprint(mol, factory) for mol in mols]
        size = 4096
        X = np.zeros((len(mols), size))
        for i, fp in enumerate(fps):
            for k, v in fp.GetNonzeroElements().items():
                idx = k % size
                X[i, idx] = v
        return X

    def substructure(self, mols, list_of_SMARTS):
        match = [any([mol.HasSubstructMatch(Chem.MolFromSmarts(subst)) for subst in list_of_SMARTS
                      if Chem.MolFromSmarts(subst)]) for mol in mols]
        return np.array(match)

    @classmethod
    def ecfp_from_mols(cls, mols):
        fps = [AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp
