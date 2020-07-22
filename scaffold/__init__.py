# coding=utf-8
import inspect
import logging
import sys
from typing import Callable

import utils
from scaffold.ScaffoldFilter import IdenticalMurckoScaffold, IdenticalTopologicalScaffold, CompoundSimilarity, \
    ScaffoldSimilarity, NoScaffoldFilter

allScaffoldFilter = {"IdenticalMurckoScaffold":      IdenticalMurckoScaffold,
                     "IdenticalTopologicalScaffold": IdenticalTopologicalScaffold,
                     "CompoundSimilarity":           CompoundSimilarity,
                     "ScaffoldSimilarity":           ScaffoldSimilarity,
                     "None":                         NoScaffoldFilter,
                     "NoFilter":                     NoScaffoldFilter}


def get_scaffoldfilter(name, **kwargs) -> Callable:
    """Function that initializes and returns a scoring function by name"""
    name = name.replace("_", "-")
    if name in allScaffoldFilter:
        scaffoldfilter = allScaffoldFilter[name]
        # we pass only the kwargs the scoring function knows
        argspec = inspect.getfullargspec(scaffoldfilter)
        nargs = {}
        for a in argspec.args:
            if a in kwargs:
                nargs[a] = kwargs[a]
        return scaffoldfilter(**nargs)
    else:
        logging.error("Scaffold filter must be one of {}".format([f for f in allScaffoldFilter]))
        sys.exit(1)


def get_scaffoldfilter_argparse(name):
    name = name.replace("_", "-")
    if name in allScaffoldFilter:
        scaffold_filter = allScaffoldFilter[name]
        parser = utils.UnderscoreArgumentParser(prog=name, usage="{}".format(name),
                                                description=inspect.getdoc(scaffold_filter), add_help=False)
        # we extract all kwargs the scoring function knows
        return utils.add_kwargs_to_parser(scaffold_filter, parser)
    else:
        logging.error("Scaffold filter must be one of {}".format([f for f in allScaffoldFilter]))
        sys.exit(1)
