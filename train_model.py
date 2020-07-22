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

rdBase.DisableLog('rdApp.error')


def save_model(model, model_path, epoch, save_each_epoch):
    model.checkpoint()

    path = model_path
    if save_each_epoch:
        path += ".{}".format(epoch)

    model.save(path)


def train(model: models.reinvent.Model, smiles_list: List[str], model_path: str, epochs=10, lr=0.001, patience=30000,
          batch_size=128, steps_to_change_lr=500, lr_change=0.01, save_each_epoch=False,
          temperature=1.0):
    """
    Trains a model
    :param model: the model to train
    :param smiles_list: a list of SMILES to train on
    :param model_path: path where to save the model
    :param epochs: number of epochs to train
    :param lr: Learning rate for the optimizer
    :param patience: number of steps until the early stop kicks in and interrupts the training
    :param batch_size: Batch size of the model
    :param temperature: Factor by which which the logits are dived. Small numbers make the model more confident on each
                        position, but also more conservative. Large values result in random predictions at each step.
    :return:
    """
    cuda_is_available = torch.cuda.is_available()
    if cuda_is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        logging.info("CUDA not available. training will be slow")
    
    model.to(device)
    # Create a Dataset from a SMILES file
    moldata = Dataset.for_model(smiles_list, model)

    print("batch size: {}\n".format(batch_size))
    data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=Dataset.collate_fn)

    # we stop early if the loss does not change significantly anymore
    lowest_loss = np.float("inf")
    eps = 0.01
    overall_patience = patience
    patience = overall_patience

    optimizer = torch.optim.Adam(model.rnn.parameters(), lr=lr)
    for epoch in range(epochs):
        logging.info("Start Epoch {}".format(epoch))
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            if cuda_is_available:
                seqs = batch.cuda(device, non_blocking=True)
            else:
                seqs = batch.to(device)

            # Calculate loss
            log_p, _ = model.likelihood(seqs, temperature=temperature)
            loss = - log_p.mean()
            if loss.item() + eps < lowest_loss:
                patience = overall_patience
                lowest_loss = loss.item()
            else:
                patience -= 1

            if patience == 0:
                tqdm.write("*************Epoch {:2d}****************".format(epoch))
                tqdm.write("*** NO LOSS IMPROVEMENT AT STEP {:3d} ***".format(step))
                tqdm.write("*************EARLY  STOP****************")
                # Save the Prior
                save_model(model, model_path, epoch, save_each_epoch)
                return

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % (steps_to_change_lr // max(1, torch.cuda.device_count())) == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=lr_change)
                tqdm.write(("Epoch {:3d}   step {:3d}    loss: {:5.2f}    "
                            "patience: {}    lr: {}").format(epoch, step,
                                                             loss.data[0],
                                                             patience,
                                                             optimizer.param_groups[0]["lr"]))
                seqs, likelihood, _ = model.sample(128, temperature=temperature)
                valid = 0
                tqdm.write("\n\n*************Epoch {:2d}****************".format(epoch))
                smiles = model.sequence_to_smiles(seqs)
                for i, smile in enumerate(smiles):
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("****************************************\n")

        # Save the model after each epoch
        save_model(model, model_path, epoch, save_each_epoch)


def main():
    parser = argparse.ArgumentParser(description="Train a model on a SMILES file")
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
    parser.add_argument("--epochs", help="Number of epochs to train [DEFAULT: 10]", type=int, default=10)
    parser.add_argument("--batch-size", help="Number of molecules processed per batch [DEFAULT: 128]", type=int,
                        default=128)
    parser.add_argument("--lr", help="Learning rate for training [DEFAULT: 0.001]", type=float, default=0.001)
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
