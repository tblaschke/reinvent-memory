# coding=utf-8

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem

import utils


class external_script(object):
    """
    Score the molecules with an external script. This scoring function writes the smiles list into a file and calls
    the defined script and reads it's output file after its finished."""

    def __init__(self, script: utils.FilePath, pass_only_valid_mols=True, continue_without_score=False,
                 use_predictable_filenames=False):

        if shutil.which(script) is None and shutil.which("./" + script) is None:
            raise ValueError("Can't execute {}. Maybe no executable permission?".format(script))

        if use_predictable_filenames:
            if "SLURM_JOB_ID" in os.environ:
                id_ = os.environ["SLURM_JOB_ID"]
            else:
                id_ = os.getpid()
            scored_file: utils.FilePath = "scoredsmiles_{}.smi".format(id_)
            smiles_to_score_file: utils.FilePath = "smilestoscore_{}.smi".format(id_)
        else:
            _, scored_file = tempfile.mkstemp(prefix='scoredsmiles_', suffix='.smi')
            os.remove(scored_file)
            _, smiles_to_score_file = tempfile.mkstemp(prefix='smilestoscore_', suffix='.smi')
            os.remove(smiles_to_score_file)

        self.script = script + " --in_file {} --out_file {}".format(smiles_to_score_file, scored_file)
        self.scored_file = scored_file
        self.smiles_to_score_file = smiles_to_score_file
        self.pass_only_valid_mols = pass_only_valid_mols
        self.continue_without_score = continue_without_score

    def __call__(self, smiles: List[str]) -> dict:
        if self.pass_only_valid_mols:
            mols = [Chem.MolFromSmiles(smile) for smile in smiles]
            valid = [1 if mol != None else 0 for mol in mols]
            valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
            valid_mols = [mols[idx] for idx in valid_idxs]

            valid_mols_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in valid_mols]
            df = {"ID": valid_idxs, "SMILES": valid_mols_smiles}
        else:
            df = {"ID": list(range(len(smiles))), "SMILES": smiles}
        df = pd.DataFrame(df)
        df = df[["SMILES", "ID"]]  # reorder the columns
        df.to_csv(self.smiles_to_score_file, index=False, sep=" ", header=None)
        return_code = subprocess.call(self.script, shell=True)
        if return_code != 0:
            logging.warning(
                "{} returned non-zero exit status {}. Undefined behaviour may occur.".format(self.script,
                                                                                             return_code))
        if not os.path.exists(self.scored_file):
            if not self.continue_without_score:
                logging.error("Can't find {}. If this is an expected behaviour of your script use the "
                              "--continue_without_score=True option".format(self.scored_file))
                raise RuntimeError("Can't find scored_file")
            else:
                logging.info("No scored_file found. Continue with 0 score")
                return {"total_score": np.full(len(smiles), 0, dtype=np.float32)}

        with open(self.scored_file, 'r') as f:
            firstline = f.readline().strip().split()

        if "ID" in firstline:
            header = 0
            if "total_score" in firstline:
                names = firstline
            elif "Score" or "score" in firstline:
                names = list(map(lambda x: "total_score" if x == "Score" or x == "score" else x, firstline))
            else:
                logging.error("Found ID column but no score column")
                raise RuntimeError("Found ID column but no score column")
        else:
            logging.debug("Could not find a header line. Assume ID in the first column and total_score in the second")
            header = None
            names = ["ID", "total_score"] + ["score_{}".format(i) for i in range(1, len(firstline) - 1)]
        df_scores = pd.read_csv(self.scored_file, sep="\s+", header=header, names=names)
        df_scores.index = df_scores["ID"]
        df_scores.drop("ID", axis=1, inplace=True)

        # we add a 0.0 score for all missing IDs
        for i in range(len(smiles)):
            if i not in df_scores.index:
                df_scores.loc[i] = {k: 0.0 for k in df_scores.keys()}
        df_scores.sort_index(inplace=True)

        # clean up the files
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.scored_file)
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.smiles_to_score_file)
        return {k: np.array(df_scores[k], dtype=np.float32) for k in df_scores.keys()}
