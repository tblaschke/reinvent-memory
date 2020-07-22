# coding=utf-8

import argparse
import logging
from typing import List

import numpy as np
import torch
from rdkit import Chem
from rdkit import rdBase
from torch.utils.data import DataLoader
from tqdm import tqdm

import models.reinvent
from chem import smiles
from models.reinvent.dataset import Dataset
from utils import decrease_learning_rate
from train_model import train

rdBase.DisableLog('rdApp.error')


def save_model(model, model_path, epoch, save_each_epoch):
    model.checkpoint()

    path = model_path
    if save_each_epoch:
        path += ".{}".format(epoch)

    model.save(path)


def main():
    parser = argparse.ArgumentParser(description="Performs transfer learning of a model on a SMILES file")
    parser.add_argument("--input-model", '-i', help='Prior model file',
                        type=str, required=True)
    parser.add_argument("--output-model", '-o', help='Path to the output model',
                        type=str, required=True)
    parser.add_argument("--input-smiles", '-s', help='Path to the SMILES file',
                        type=str, required=True)
    parser.add_argument("--standardize-smiles", help='Set if want to standardize the SMILES using RDKIT',
                        action="store_true", default=False)
    parser.add_argument("--save-each-epoch", help="Set to save each epoch in a different model file.",
                        action="store_true", default=False)
    parser.add_argument("--steps-to-change-lr", "--sclr", help="Number of steps to change learning rate", type=int,
                        default=500)
    parser.add_argument("--lr-change", "--lrc", help="Ratio which the learning rate is changed", type=float,
                        default=0.01)
    parser.add_argument("--epochs", help="Number of epochs to train [DEFAULT: 20]", type=int, default=20)
    parser.add_argument("--batch-size", help="Number of molecules processed per batch [DEFAULT: 128]", type=int,
                        default=128)
    parser.add_argument("--lr", help="Learning rate for training [DEFAULT: 0.0005]", type=float, default=0.0005)
    parser.add_argument("--patience",
                        help=("Number of steps where the training get stopped if no loss improvement is noticed. "
                              "[DEFAULT: 30000]"),
                        type=int, default=30000)
    parser.add_argument("--temperature", "-t",
                        help=("Temperature for the sequence sampling. Has to be larger than 0. Values below 1 make "
                              "the RNN more confident in it's generation, but also more conservative. "
                              "Values larger than 1 result in more random sequences. [DEFAULT: 1.0]"),
                        type=float, default=1.0)
    args = parser.parse_args()

    # setup the logger to get a nice output
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s',
                        datefmt='%H:%M:%S'
                        )
    model = models.reinvent.Model.load_from_file(args.input_model)

    logging.info("Reading smiles...")
    with open(args.input_smiles, 'r') as f:
        lines = [line.strip().split()[0] for line in f]
    logging.info("Read {} lines".format(len(lines)))
    if args.standardize_smiles:
        logging.info("Standardize SMILES")
        smiles_list = smiles.standardize_smiles_list(lines)
    else:
        smiles_list = lines

    train(model, smiles_list, model_path=args.output_model, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, patience=args.patience, save_each_epoch=args.save_each_epoch,
          steps_to_change_lr=args.steps_to_change_lr, lr_change=args.lr_change, temperature=args.temperature)


if __name__ == "__main__":
    main()
