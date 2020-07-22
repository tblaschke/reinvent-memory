# coding=utf-8

from typing import List

import numpy as np
import openeye.oechem as oechem
import openeye.oedocking as oedocking
import openeye.oeomega as oeomega

import utils


class docking_base(object):
    def __init__(self, receptor: utils.FilePath):
        self.receptor_file = receptor
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        self.omega = oeomega.OEOmega(omegaOpts)
        oechem.OEThrow.SetLevel(10000)
        oereceptor = oechem.OEGraphMol()
        oedocking.OEReadReceptorFile(oereceptor, self.receptor_file)
        self.dock = oedocking.OEDock()
        self.dock.Initialize(oereceptor)

    def __call__(self, smile):
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, smile):
            return 0.0
        if self.omega(mol):
            dockedMol = oechem.OEGraphMol()
            self.dock.DockMultiConformerMolecule(dockedMol, mol)
            score = dockedMol.GetEnergy()
            score = max(0.0, -(score + 8) / 10)
            return score

    def __reduce__(self):
        """
        :return: A tuple with the constructor and its arguments. Used to reinitialize the object for pickling
        """
        return docking_base, (self.receptor_file,)


class docking(docking_base):
    """Scores based on the omega docking."""
    def __call__(self, smiles: List[str]) -> dict:
        score = np.full(len(smiles), 0, dtype=np.float32)
        for idx, smi in enumerate(smiles):
            score[idx] = super()(smi)
        return {"total_score": np.array(score, dtype=np.float32)}
