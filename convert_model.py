# coding=utf-8
import logging

import torch

import utils
from models.reinvent import Vocabulary, Model


def main():
    parser = utils.UnderscoreArgumentParser(
        description="Converts the old model format with separate vocabulary file to the "
                                                 "new single file format")
    parser.add_argument("--input-model", '-i', help='The model.ckpt to read',
                        type=str, required=True)
    parser.add_argument("--vocabulary", "-v", help='The used vocabulary',
                        type=str, required=True)
    parser.add_argument("--output-model", '-o', help='Path were to save the new model',
                        type=str, required=True)
    args = parser.parse_args()

    # setup the logger to get a nice output
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s',
                        datefmt='%H:%M:%S'
                        )

    voc = Vocabulary()
    voc.init_from_voc_file(args.vocabulary)
    initial_weights = torch.load(args.input_model)
    model = Model(voc, initialweights=initial_weights)
    model.save(args.output_model)
    logging.info("Model saved. You can now remove the old model and vocabulary")


if __name__ == "__main__":
    main()
