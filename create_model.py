# coding=utf-8
import argparse
import logging

from chem.smiles import standardize_smiles_list
from models import reinvent
from utils import write_smiles_to_file


def main():
    parser = argparse.ArgumentParser(description="Creates a new untrained model from a SMILES file")
    parser.add_argument("--input-smiles", '-i', help='The smi filepath used to create the prior',
                        type=str, required=True)
    parser.add_argument("--standardize-smiles", "-s", help='Set to true if want to standardize the SMILES using RDKIT',
                        action="store_true", default=False)
    parser.add_argument("--save-file", '-o', help='The filepath were to save to prior',
                        type=str, required=True)
    parser.add_argument("--save-standardized-smiles", '-smi',
                        help='The filepath were to save the standardized smiles (optional, recommended)',
                        type=str, required=False, default=None)
    args = parser.parse_args()

    # setup the logger to get a nice output
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s',
                        datefmt='%H:%M:%S'
                        )

    logging.info("Reading smiles...")
    with open(args.input_smiles, 'r') as f:
        lines = [line.strip().split(" ")[0] for line in f]
    logging.info("Read {} lines".format(len(lines)))
    if args.standardize_smiles:
        logging.info("Standardize SMILES")
        smiles_list = standardize_smiles_list(lines)
        if args.save_standardized_smiles:
            logging.info("Write SMILES to {}".format(args.save_standardized_smiles))
            write_smiles_to_file(smiles_list, args.save_standardized_smiles)
    else:
        smiles_list = lines
    voc = reinvent.Vocabulary()
    logging.info("Build vocabulary")
    voc.init_from_smiles_list(smiles_list)

    prior = reinvent.Model(voc=voc)
    logging.info("Save prior at {}".format(args.save_file))
    prior.save(args.save_file)


if __name__ == "__main__":
    main()
