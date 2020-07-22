# coding=utf-8

import argparse
import logging

from models import reinvent
from utils import write_smiles_to_file


def main():
    parser = argparse.ArgumentParser(description="Sample SMILES from a model")
    parser.add_argument("--model", '-m', help='Path to the model',
                        type=str, required=True)
    parser.add_argument("--number", '-n', help='Number of SMILES to sample',
                        type=int, required=False, default=500)
    parser.add_argument("--save-smiles", '-o',
                        help='The filepath the SMILES are saved',
                        type=str, required=False, default=None)
    parser.add_argument("--temperature", "-t",
                        help=("Temperature for the sequence sampling. Has to be larger than 0. Values below 1 make "
                              "the RNN more confident in it's generation, but also more conservative. Values larger "
                              "than 1 result in more random sequences. [DEFAULT: 1.0]"),
                        type=float, default=1.0)
    args = parser.parse_args()

    # setup the logger to get only error output
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s',
                        datefmt='%H:%M:%S'
                        )

    model = reinvent.Model.load_from_file(args.model)
    samples, _ = model.sample_smiles(args.number, temperature=args.temperature)

    if args.save_smiles:
        write_smiles_to_file(samples, args.save_smiles)
    else:
        for s in samples:
            print(s)


if __name__ == "__main__":
    main()
