# coding=utf-8

import logging
from typing import List

import numpy as np
import openeye.oechem as oechem
import openeye.oeomega as oeomega
import openeye.oeshape as oeshape

import utils
from .slurmmanager import slurmmanager


class rocs_similarity_base(object):
    def __init__(self, ligand: utils.FilePath, max_tanimoto=0.6, shape_weight=0.5, color_weight=0.5):
        self.ligand = ligand
        self.k = max_tanimoto
        self.shape_weight = shape_weight
        self.color_weight = color_weight
        reffs = oechem.oemolistream(self.ligand)

        refmol = oechem.OEMol()
        oechem.OEReadMolecule(reffs, refmol)
        self.best = oeshape.OEBestOverlay()
        self.best.SetRefMol(refmol)
        self.best.SetColorForceField(oeshape.OEColorFFType_ImplicitMillsDean)
        self.best.SetColorOptimize(True)
        self.best.SetInitialOrientation(oeshape.OEBOOrientation_Inertial)
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        self.omega = oeomega.OEOmega(omegaOpts)
        self.keepsize = 1
        oechem.OEThrow.SetLevel(10000)

    def __call__(self, smile):
        imol = oechem.OEMol()
        if not oechem.OESmilesToMol(imol, smile):
            return 0
        best_Tanimoto = 0.0
        if self.omega(imol):
            scoreiter = oeshape.OEBestOverlayScoreIter()
            oeshape.OESortOverlayScores(scoreiter, self.best.Overlay(imol),
                                        oeshape.OEHighestTanimotoCombo())
            for score in scoreiter:
                outmol = oechem.OEGraphMol(imol.GetConf(oechem.OEHasConfIdx(score.fitconfidx)))
                score.Transform(outmol)
                best_Tanimoto = (self.shape_weight * score.GetTanimoto()) + (
                        self.color_weight * score.GetColorTanimoto())
                best_Tanimoto = np.minimum(best_Tanimoto, self.k)
                break

        else:
            logging.debug("Omega failed")
        return best_Tanimoto

    def get_conformation(self, smile):
        imol = oechem.OEMol()
        if not oechem.OESmilesToMol(imol, smile):
            return None
        if self.omega(imol):
            scoreiter = oeshape.OEBestOverlayScoreIter()
            oeshape.OESortOverlayScores(scoreiter, self.best.Overlay(imol),
                                        oeshape.OEHighestTanimotoCombo())
            for score in scoreiter:
                outmol = oechem.OEGraphMol(imol.GetConf(oechem.OEHasConfIdx(score.fitconfidx)))
                score.Transform(outmol)
                ofs = oechem.oemolostream()
                ofs.openstring()
                ofs.SetFormat(oechem.OEFormat_MOL2)
                oechem.OEWriteMolecule(ofs, outmol)
                result = ofs.GetString().decode()
                return result


class rocs_similarity(rocs_similarity_base):
    """Scores based on ROCS shape and color similarity. Runs on a single CPU core."""
    def __call__(self, smiles: List[str]) -> dict:
        score = np.full(len(smiles), 0, dtype=np.float32)
        for idx, smi in enumerate(smiles):
            score[idx] = super().__call__(smi)
        return {"total_score": np.array(score, dtype=np.float32)}

    def __reduce__(self):
        """
        :return: A tuple with the constructor and its arguments. Used to reinitialize the object for pickling
        """
        return rocs_similarity, (self.ligand, self.k, self.shape_weight, self.color_weight)


class rocs_similarity_slurm(slurmmanager):
    """Scores based on ROCS shape and color similarity. Distributes the calculation using SLURM."""

    def __init__(self, ligand: utils.FilePath, port=31992, nb_local=8, nb_slurm=6, cpu_per_job=8, max_tanimoto=0.6,
                 shape_weight=0.5, color_weight=0.5):
        super().__init__(port=port, nb_local=nb_local, nb_slurm=nb_slurm, cpu_per_job=cpu_per_job, ligand=ligand,
                         max_tanimoto=max_tanimoto, shape_weight=shape_weight, color_weight=color_weight,
                         scoring_function="rocs_similarity")

    def __call__(self, smiles: List[str]) -> dict:
        return {"total_score": super().__call__(smiles)}
