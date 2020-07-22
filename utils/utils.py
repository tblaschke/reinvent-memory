# coding=utf-8

import argparse
import inspect
import os
import signal
from typing import Callable, NewType

import numpy as np
import torch
from rdkit import Chem


def Variable(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def fraction_valid_smiles(smiles):
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


FilePath = NewType("FilePath", str)
SMILES = NewType("SMILES", str)


def format_help_for_epilog(argparser, prefix):
    formatter = argparser._get_formatter()

    # usage
    formatter.add_usage(argparser.usage, argparser._actions,
                        argparser._mutually_exclusive_groups, prefix=prefix)

    # indent the formatter
    formatter._indent()
    formatter._indent()
    formatter._indent()

    # description
    formatter.add_text(argparser.description)

    # positionals, optionals and user-defined groups
    for action_group in argparser._action_groups:
        formatter.start_section(action_group.title)
        formatter.add_text(action_group.description)
        formatter.add_arguments(action_group._group_actions)
        formatter.end_section()

    # epilog
    formatter.add_text(argparser.epilog)

    help_text = formatter.format_help()
    # determine help from format above
    return help_text


class UnderscoreArgumentParser(argparse.ArgumentParser):
    def _parse_optional(self, arg_string):
        # if it's an empty string, it was meant to be a positional
        if not arg_string:
            return None

        # if it doesn't start with a prefix, it was meant to be positional
        if not arg_string[0] in self.prefix_chars:
            return None

        # if the option string is present in the parser, return the action
        if arg_string in self._option_string_actions:
            action = self._option_string_actions[arg_string]
            return action, arg_string, None

        if arg_string.replace("_", "-") in self._option_string_actions:
            action = self._option_string_actions[arg_string.replace("_", "-")]
            return action, arg_string.replace("_", "-"), None

        # if it's just a single character, it was meant to be positional
        if len(arg_string) == 1:
            return None

        # if the option string before the "=" is present, return the action
        if '=' in arg_string:
            option_string, explicit_arg = arg_string.split('=', 1)
            if option_string in self._option_string_actions:
                action = self._option_string_actions[option_string]
                return action, option_string, explicit_arg
            if option_string.replace("_", "-") in self._option_string_actions:
                action = self._option_string_actions[option_string.replace("_", "-")]
                return action, option_string.replace("_", "-"), explicit_arg

        if self.allow_abbrev:
            # search through all possible prefixes of the option string
            # and all actions in the parser for possible interpretations
            option_tuples = self._get_option_tuples(arg_string)

            # if multiple actions match, the option string was ambiguous
            if len(option_tuples) > 1:
                options = ', '.join([option_string
                                     for action, option_string, explicit_arg in option_tuples])
                args = {'option': arg_string, 'matches': options}
                msg = ('ambiguous option: %(option)s could match %(matches)s')
                self.error(msg % args)

            # if exactly one action matched, this segmentation is good,
            # so return the parsed action
            elif len(option_tuples) == 1:
                option_tuple, = option_tuples
                return option_tuple

        # if it was not found as an option, but it looks like a negative
        # number, it was meant to be positional
        # unless there are negative-number-like options
        if self._negative_number_matcher.match(arg_string):
            if not self._has_negative_number_optionals:
                return None

        # if it contains a space, it was meant to be a positional
        if ' ' in arg_string:
            return None

        # it was meant to be an optional but there is no such option
        # in this parser (though it might be a valid option in a subparser)
        return None, arg_string, None


def add_kwargs_to_parser(function: Callable, parser: argparse.ArgumentParser):
    """
    This function extracts all keyword-only arguments of a function and creates a argparser
    with all keyword arguments.
    It adds the default value to the argparser. If the default value is None then no default value is added.
    If a annotation exists it;s add this as a type information.
    """
    argspec = inspect.getfullargspec(function)
    requiredArgs = parser.add_argument_group('required arguments')
    optionalArgs = parser.add_argument_group('optional arguments')
    if argspec.defaults:
        nb_nondefault = len(argspec.args) - len(argspec.defaults)
    else:
        nb_nondefault = len(argspec.args)
    for idx, arg in enumerate(argspec.args):
        if arg == "self":
            continue
        # lets check if the argument has a default and is therefore optional
        default_idx = idx - nb_nondefault
        if default_idx >= 0:
            arg_default = argspec.defaults[default_idx]
            if arg in argspec.annotations:
                arg_type = argspec.annotations[arg]
            else:
                arg_type = type(arg_default)

            optionalArgs.add_argument("--{}".format(arg.replace("_", "-")), required=False,
                                      default=arg_default,
                                      type=arg_type,
                                      metavar="<{}>".format(str(arg_type.__name__)))
        else:
            if arg in argspec.annotations:
                arg_type = argspec.annotations[arg]
                requiredArgs.add_argument("--{}".format(arg.replace("_", "-")), required=True,
                                          type=arg_type,
                                          metavar="<{}>".format(str(arg_type.__name__)))
            else:
                requiredArgs.add_argument("--{}".format(arg.replace("_", "-")), required=True, metavar="")
    return parser


def NLLLoss(inputs, targets):
    """
    Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

    :param inputs: (batch_size, num_classes) *Log probabilities of each class*
    :param targets: (batch_size) *Target class index*
    :return: loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss


class Timeout:
    """Class for setting a time limit on a function. Used when pharmacophoric fingerprints
       of certain structures take too long."""

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds if seconds > 0 else 0
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def write_smiles_to_file(smiles_list, fname):
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


def read_smi_file(file_path):
    with open(file_path, "r") as smi_file:
        return [smi.rstrip().split()[0] for smi in smi_file]


def find_dir_suffix(path: FilePath, minnb: int = 1):
    if os.path.exists(path):
        path = path + "_{}"
        while os.path.exists(path.format(minnb)):
            minnb += 1
        return path.format(minnb)
    else:
        return path
