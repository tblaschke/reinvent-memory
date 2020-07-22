# coding=utf-8
import logging
import re

import numpy as np


class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ['$', '^']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_voc_file(init_from_file)

    def __eq__(self, other):
        if isinstance(other, Vocabulary):
            if other.vocab == self.vocab:
                return True
        return False

    def ensure_new_special_tokens(self):
        """
        The old Vocabulary used 'GO' and 'EOS'. Now we are using '^' and '$'. This functions replaces the old token
        with the new ones if necessary
        :return:
        """

        def replace_char_in_vocabs(original, replacement):
            tokennum = self.vocab[original]
            self.vocab[replacement] = tokennum
            self.reversed_vocab[tokennum] = replacement
            del self.vocab[original]

        if "GO" in self.vocab and "^" not in self.vocab:
            replace_char_in_vocabs("GO", "^")
        if "EOS" in self.vocab and "$" not in self.vocab:
            replace_char_in_vocabs("EOS", "$")
        for i, token in enumerate(self.chars):
            if token == "GO":
                self.chars[i] = "^"
            elif token == "EOS":
                self.chars[i] = "$"

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['$']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles, without_eos=False):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = r"(\[[^\[\]]{1,6}\])"
        smiles = self.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        if not without_eos:
            tokenized.append('$')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_voc_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)
        logging.debug("Initialized with vocabulary: {}".format(chars))

    def save_to_voc_file(self, file):
        with open(file, 'w') as f:
            for char in self.chars:
                f.write(char + "\n")

    def init_from_smiles_list(self, smiles_list):
        """ Adds all characters present in a list of SMILES.
         Uses regex to find characters/tokens of the format '[x]'.
        Initialize
        :param smiles_list: list of SMILES
        :return:
        """
        add_chars = set()
        for smiles in smiles_list:
            tokens = self.tokenize(smiles, without_eos=True)
            add_chars |= set(tokens)
        add_chars = list(add_chars)
        logging.debug("Add the following character to the vocabulary: {}".format(add_chars))
        self.add_characters(add_chars)

    @classmethod
    def replace_halogen(cls, string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        string = br.sub('R', string)
        string = cl.sub('L', string)

        return string

    def filter_smiles_list(self, smiles_list):
        """Filters SMILES on the characters they contain.
           Used to remove SMILES containing characters we cannot encode."""
        smiles_list_valid = []
        for smiles in smiles_list:
            tokenized = self.tokenize(smiles)
            if all([char in self.chars for char in tokenized]):
                smiles_list_valid.append(smiles)
        return smiles_list_valid
