# coding=utf-8

import glob
import inspect
import logging
import sys
from os.path import dirname, basename, isfile
from typing import List, Callable

import utils

"""
Let's have a little bit of black magic here. We import all .py files in our module folder and take 
every callable object which has a signature (self, smiles) as a possible scoring function.
The arguments in __init__ get be requested and dynamically added to an argument parser
"""

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a list of
   smiles as an argument and returns a dictionary all (component)scores . The "total_score" is necessary and must be  
   matching np.array of floats."""


def _isScoringFunction(obj):
    """Return True is object is a valid scoring function"""
    if inspect.isclass(obj):
        if callable(obj):
            sig = inspect.getfullargspec(obj.__call__)
            if len(sig.args) == 2:
                if "return" in sig.annotations and 'smiles' in sig.annotations:
                    if sig.annotations['return'] == dict and sig.annotations['smiles'] == List[str]:
                        return True
    return False


modules = glob.glob(dirname(__file__) + "/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

allScoringFunctions = {}
# import all modules we found and generate a collection of all valid scoring functions
for module in __all__:
    __import__(__package__ + "." + module, {}, {}, [], 0)
    all_valid_members = inspect.getmembers(globals()[module], _isScoringFunction)
    for name, func in all_valid_members:
        name = name.replace("_", "-")
        if name in allScoringFunctions:
            if allScoringFunctions[name] != func:
                logging.warning(
                        ("Scoring function {} is defined twice. Old definition in {}."
                         " Using new in {}").format(name, allScoringFunctions[name], func))
        allScoringFunctions[name] = func


def get_scoring_function(name, **kwargs) -> Callable:
    """Function that initializes and returns a scoring function by name"""
    name = name.replace("_", "-")
    if name in allScoringFunctions:
        scoring_function = allScoringFunctions[name]
        # we pass only the kwargs the scoring function knows
        argspec = inspect.getfullargspec(scoring_function)
        nargs = {}
        for a in argspec.args:
            if a in kwargs:
                nargs[a] = kwargs[a]
        return scoring_function(**nargs)
    else:
        logging.error("Scoring function must be one of {}".format([f for f in allScoringFunctions]))
        sys.exit(1)


def get_scoring_argparse(name):
    name = name.replace("_", "-")
    if name in allScoringFunctions:
        scoring_function = allScoringFunctions[name]
        parser = utils.UnderscoreArgumentParser(prog=name, usage="{}".format(name), description=inspect.getdoc(
                scoring_function), add_help=False)
        # we extract all kwargs the scoring function knows
        return utils.add_kwargs_to_parser(scoring_function, parser)
    else:
        logging.error("Scoring function must be one of {}".format([f for f in allScoringFunctions]))
        sys.exit(1)
