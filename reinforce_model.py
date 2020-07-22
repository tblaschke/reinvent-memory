# coding=utf-8

import argparse
import contextlib
import json
import logging
import os
import sys
import time

import models.reinvent
import reinforcement
import scaffold
import scoring
from utils import format_help_for_epilog, UnderscoreArgumentParser, FilePath, find_dir_suffix


def main():
    strtime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

    logging.basicConfig(level=logging.DEBUG)

    fmt = logging.Formatter(
        fmt='%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S')
    for handler in logging.getLogger().handlers:
        handler.setFormatter(fmt)

    _scoring_help = "\n".join([format_help_for_epilog(scoring.get_scoring_argparse(name), prefix=" scoring: ")
                               for name in sorted(scoring.allScoringFunctions)]) + "\n"
    _scoring_help += "\n".join([format_help_for_epilog(scaffold.get_scaffoldfilter_argparse(name), prefix=" filter: ")
                                for name in sorted(scaffold.allScaffoldFilter)])

    parser = UnderscoreArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False,
                                      epilog=_scoring_help)

    requiredArgs = parser.add_argument_group('required arguments')
    optionalArgs = parser.add_argument_group('optional arguments')
    optionalArgs.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                              help='show this help message and exit')

    requiredArgs.add_argument("--scoring-function",
                              help='Scoring function to use. Allowed values are: ' +
                                   ', '.join(sorted(scoring.allScoringFunctions.keys())),
                              metavar="<scoring>", type=str, required=True,
                              choices=sorted(list(set(
                                      list(scoring.allScoringFunctions.keys()) + [name.replace("-", "_") for name in
                                                                                  scoring.allScoringFunctions.keys()]))))
    requiredArgs.add_argument("--scaffold-filter",
                              help='Scaffold filter to use. Allowed values are: ' +
                                   ', '.join(sorted(scaffold.allScaffoldFilter.keys())),
                              metavar="<filter>", type=str, required=True,
                              choices=sorted(list(set(
                                      list(scaffold.allScaffoldFilter.keys()) + [name.replace("-", "_") for name in
                                                                                 scaffold.allScaffoldFilter.keys()]))))
    optionalArgs.add_argument("--name", help="Name of the experiment. Default: if no name is provided and the "
                                             "script is running within SLURM it uses the name provided by "
                                             "SLURM_JOB_NAME otherwise noname",
                              type=str,
                              default=None,
                              metavar="<str>")
    optionalArgs.add_argument("--description", help="Description of the experiment. Currently just used in "
                                                    "Vizor. Default N/A", type=str,
                              default="N/A", metavar="<str>")

    optionalArgs.add_argument("--prior", help='Prior to use. Default priors/ChEMBL/Prior.ckpt', type=str,
                              default='priors/ChEMBL/Prior.ckpt', metavar="<{}>".format(str(FilePath.__name__)))
    optionalArgs.add_argument("--agent", help='Agent to use. If None the agent is initialized from the prior.',
                              type=str, default='None', metavar="<{}>".format(str(FilePath.__name__)))

    optionalArgs.add_argument("--steps", help='Iterations to run. Default: 500', type=int, default=500, metavar="<int>")
    optionalArgs.add_argument("--reset", help="Number of iteration after which the Agent is reset after the first "
                                              "time the average score is above reset-cutoff-score."
                                              "Default 0 (not active)",
                              type=int, default=0, metavar="<int>")
    optionalArgs.add_argument("--reset-cutoff-score", help="Average Score which have to be reached to start the "
                                                           "reset countdown of the Agent. Default 0.6",
                              type=float, default=0.6, metavar="<float>")
    optionalArgs.add_argument("--sigma", help='Scoring Sigma. Default: 120', type=float, default=120, metavar="<int>")

    optionalArgs.add_argument("--temperature", "-t",
                              help=("Temperature for the sequence sampling. Has to be larger than 0. "
                                    "Values below 1 make the RNN more confident in it's generation, "
                                    "but also more conservative. Values larger than 1 result in more random sequences. "
                                    "[DEFAULT: 1.0]"),
                              type=float, default=1.0, metavar="<float>")

    optionalArgs.add_argument("--debug", "-v", help='Verbose messages', action='store_true', default=False)
    optionalArgs.add_argument("--noteset", "-vv", help='More verbose messages', action='store_true', default=False)

    optionalArgs.add_argument("--experience", help='Enable experience replay. Default False', type=bool,
                              default=False, metavar="<bool>")
    optionalArgs.add_argument("--lr", help='Optimizer learning rate. Default: 0.0001', type=float, default=0.0001,
                              metavar="<float>")
    optionalArgs.add_argument("--batch-size", help='How many compounds are sampled per step. Default: 128', type=int,
                              default=128, metavar="<int>")

    optionalArgs.add_argument("--logdir",
                              help="Dictionary to save the log. Default ~/REINVENT/logs/<name>",
                              type=str, metavar="<{}>".format(str(FilePath.__name__)),
                              default=None)
    optionalArgs.add_argument("--resultdir",
                              help="Dictionary to save the results.  Default ~/REINVENT/results/<name>",
                              type=str, metavar="<{}>".format(str(FilePath.__name__)),
                              default=None)
    optionalArgs.add_argument("--seed",
                              help="Set a seed for running the reinforcement learning.  [Default: NONE]",
                              type=int, metavar="<int>", default=None)

    args, extra_args = parser.parse_known_args()

    if args.seed:
        import torch
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import numpy as np
        np.random.seed(args.seed)
        import random
        random.seed(args.seed)

    # Setup the name
    if args.name is None:
        if "SLURM_JOB_NAME" in os.environ:
            args.name = os.environ["SLURM_JOB_NAME"]
        else:
            args.name = "noname"

    # Setup the logdir and resultdir
    if args.logdir is None:
        args.logdir = os.path.join(os.path.expanduser('~'), "REINVENT/logs/{}".format(args.name))
    if args.resultdir is None:
        args.resultdir = os.path.join(os.path.expanduser('~'), "REINVENT/results/{}".format(args.name))

    args.logdir = os.path.normpath(args.logdir)
    args.resultdir = os.path.normpath(args.resultdir)

    if os.path.exists(args.logdir):
        new_logdir = find_dir_suffix(args.logdir)
        logging.info("Logdir already exists. Using {} instead".format(new_logdir))
        args.logdir = new_logdir
    if os.path.exists(args.resultdir):
        new_resultdir = find_dir_suffix(args.resultdir)
        logging.info("Resultdir already exists. Using {} instead".format(new_resultdir))
        args.resultdir = new_resultdir

    os.makedirs(args.logdir)
    os.makedirs(args.resultdir)

    # Set up the logging
    fh = logging.FileHandler(os.path.join(args.logdir, 'output.log'))
    fh.setLevel(logging.INFO)
    dh = logging.FileHandler(os.path.join(args.logdir, 'debug.log'))
    dh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    if args.noteset:
        ch.setLevel(logging.NOTSET)
    elif args.debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    logginghandler = [fh, dh, ch]
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    for handler in logginghandler:
        handler.setFormatter(fmt)
        logging.getLogger().addHandler(handler)

    # first we get the scoring function
    scoring_parser = scoring.get_scoring_argparse(args.scoring_function)
    scoring_args, extra_args = scoring_parser.parse_known_args(extra_args)
    scoring_function = scoring.get_scoring_function(args.scoring_function, **vars(scoring_args))

    # now we get the scaffold filter
    scaffold_parser = scaffold.get_scaffoldfilter_argparse(args.scaffold_filter)
    scaffold_args, extra_args = scaffold_parser.parse_known_args(extra_args)
    scaffoldfilter = scaffold.get_scaffoldfilter(args.scaffold_filter, **vars(scaffold_args))

    # lets hope we have no arguments left. Otherwise we fail
    if len(extra_args) > 0:
        print("\n\033[91mERROR: unrecognized arguments: " + " ".join(extra_args) + "\033[0m\n")
        parser.print_help()
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(args.logdir, 'output.log'))
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(args.logdir, 'debug.log'))
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(args.logdir)
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(args.resultdir)
        exit(2)

    prior = models.reinvent.Model.load_from_file(args.prior)
    if args.agent == "None":
        agent = models.reinvent.Model.load_from_file(args.prior)
    else:
        agent = models.reinvent.Model.load_from_file(args.agent)

    metadata = {"name": args.name, "description": args.description, "date": strtime, 'arguments': sys.argv}
    metadata = json.dumps(metadata, sort_keys=True, indent=4, separators=(',', ': '))
    with open(args.logdir + "/metadata.json", 'w') as f:
        f.write(metadata + "\n")

    reinforcement.reinforcement_learning(agent=agent, prior=prior,
                                         scoring_function=scoring_function, scaffoldfilter=scaffoldfilter,
                                         n_steps=args.steps,
                                         experience_replay=args.experience, reset=args.reset,
                                         reset_score_cutoff=args.reset_cutoff_score,
                                         logdir=args.logdir, resultdir=args.resultdir,
                                         lr=args.lr, sigma=args.sigma,
                                         batch_size=args.batch_size,
                                         temperature=args.temperature)


if __name__ == "__main__":
    main()
