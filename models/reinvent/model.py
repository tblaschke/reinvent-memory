# coding=utf-8

import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Variable, NLLLoss
from .vocabulary import Vocabulary


class MultiGRU(nn.Module):
    """
    Implements a N layer GRU(M) cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, voc_size, gru_layer_size=512, num_gru_layers=3, embedding_layer_size=256):
        """
        Implements a N layer GRU(M) cell including an embedding layer and an output linear layer back to the size of the
        vocabulary
        :param voc_size: Size of the vocabulary.
        :param gru_layer_size: Size of each of the GRU layers.
        :param num_gru_layers: Number of GRU layers.
        :param embedding_layer_size: Size of the embedding layer.
        """

        super(MultiGRU, self).__init__()

        self._gru_layer_size = gru_layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_gru_layers = num_gru_layers

        self._other_layers = {"embedding": nn.Embedding(voc_size, self._embedding_layer_size)}
        self.add_module("embedding", self._other_layers["embedding"])

        # modules state_dict is bound to setattr and doesn't support lists, so the fields will be added with setattr
        layer = nn.GRUCell(self._embedding_layer_size, self._gru_layer_size)
        self.add_module("gru_1", layer)
        self._gru_layers = [layer]
        for i in range(2, self._num_gru_layers + 1):
            layer = nn.GRUCell(self._gru_layer_size, self._gru_layer_size)
            self._gru_layers.append(layer)
            self.add_module("gru_{}".format(i), layer)

        self._other_layers["linear"] = nn.Linear(self._gru_layer_size, voc_size)
        self.add_module("linear", self._other_layers["linear"])

    def forward(self, x, h):
        x = self._other_layers["embedding"](x)
        if h is None:
            h = Variable(torch.zeros(self._num_gru_layers, x.size()[0], self._gru_layer_size))

        h_out = Variable(torch.zeros(h.size()))
        for i, gru_layer in enumerate(self._gru_layers):
            x = h_out[i] = gru_layer(x, h[i])

        x = self._other_layers["linear"](x)
        return x, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(self._num_gru_layers, batch_size, self._gru_layer_size))

    def get_params(self):
        return {
            'gru_layer_size':       self._gru_layer_size,
            'num_gru_layers':       self._num_gru_layers,
            'embedding_layer_size': self._embedding_layer_size
            }


class Model:
    _version = "0.2"

    def __init__(self, voc: Vocabulary, initialweights: OrderedDict = None, rnn_params=None):
        """
        Implements the Prior and Agent RNN. Needs a Vocabulary instance in
        order to determine size of the vocabulary and index of the END token.
        :param voc: Vocabulary to use
        :param initialweights: Weights to initialize the RNN
        :param rnn_params: A dict with any of the accepted params in MultiGRU's constructor except for voc_size.
        """
        self.voc = voc

        if not isinstance(rnn_params, dict):
            rnn_params = {}

        self.rnn = MultiGRU(self.voc.vocab_size, **rnn_params)

        if torch.cuda.is_available():
            self.rnn.cuda()
        if initialweights:
            self.initialweights = copy.deepcopy(initialweights)
            self.rnn.load_state_dict(copy.deepcopy(initialweights))
        else:
            self.initialweights = copy.deepcopy(self.rnn.state_dict())

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Loads a model from a single file
        :param file: filpath as string
        :return: new instance of the RNN or None if it was not possible to load
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path)
        else:
            save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)
        if "version" in save_dict:
            model = Model(save_dict['vocabulary'],
                          initialweights=save_dict['initialweights'], rnn_params=save_dict.get("rnn_params", {}))
            model.rnn.load_state_dict(save_dict["currentweights"])
            if save_dict['version'] != Model._version:
                logging.warning(
                        ("Trying to load a model saved with version {} on version {}. "
                         "The model will be updated when saved").format(save_dict["version"], Model._version))
            model.voc.ensure_new_special_tokens()
            return model
        else:
            logging.error("Can't find any version information. Model will not be loaded.")
            raise ValueError("Version information not present in model.")

    def to(self, device: torch.device):
        self.rnn = self.rnn.to(device)
        return self

    def save(self, file):
        """
        Saves the model into a file
        :param file: Filepath as string
        :return: None
        """
        save_dict = {
            'version':        Model._version,
            'vocabulary':     self.voc,
            'initialweights': self.initialweights,
            'currentweights': self.rnn.state_dict(),
            'rnn_params':     self.rnn.get_params()
            }
        torch.save(save_dict, file)

    def reset(self):
        """
        Resets the RNN weights to the values saved in self.initalweights
        :return: None
        """
        self.rnn.load_state_dict(copy.deepcopy(self.initialweights))

    def checkpoint(self):
        """
        Set self.initalweights to the current weights of self.rnn
        :return:
        """
        self.initialweights = copy.deepcopy(self.rnn.state_dict())

    def likelihood(self, target, temperature=1.):
        """
        Retrieves the likelihood of a given sequence

        :param target: (batch_size * sequence_lenght) A batch of sequences
        :param temperature: Factor by which which the logits are dived. Small numbers make the model more confident on
                            each position, but also more conservative.
                            Large values result in random predictions at each step.
        :return:log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab["^"]
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = None  # creates zero tensor by default
        unfinished = torch.ones_like(start_token, dtype=torch.uint8)

        log_probs = Variable(torch.zeros(batch_size))
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            logits = logits / temperature
            log_prob = F.log_softmax(logits, dim=1)
            prob = F.softmax(logits, dim=1)
            log_prob = log_prob * unfinished.float()
            log_probs += NLLLoss(log_prob, target[:, step])
            entropy += -torch.sum((log_prob * prob), 1)

            EOS_sampled = (x[:, step] == self.voc.vocab['$']).unsqueeze(1)
            unfinished = torch.eq(unfinished - EOS_sampled, 1)
            if torch.sum(unfinished) == 0:
                break

        return log_probs, entropy

    def sample(self, batch_size, max_length=140, temperature=1.):
        """
        Sample a batch of sequences

        :param batch_size: Number of sequences to sample
        :param max_length: Maximum length of the sequences
        :param temperature: Factor by which which the logits are dived. Small numbers make the model more confident on
                             each position, but also more conservative. Large values result in random predictions at
                             each step.
        return:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab["^"]
        h = None  # creates zero tensor by default
        x = start_token
        unfinished = torch.ones_like(start_token, dtype=torch.uint8)
        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        entropy = Variable(torch.zeros(batch_size))

        for step in range(max_length):
            logits, h = self.rnn(x, h)
            logits = logits / temperature
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            x = torch.multinomial(prob, 1).view(-1)
            sequences.append(x.view(-1, 1))
            log_prob = log_prob * unfinished.unsqueeze(1).float()
            log_probs += NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['$'])
            unfinished = torch.eq(unfinished - EOS_sampled, 1)
            if torch.sum(unfinished) == 0:
                break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy

    def sequence_to_smiles(self, sequences):
        smiles = []
        for seq in sequences.cpu().numpy():
            smiles.append(self.voc.decode(seq))
        return smiles

    def sample_smiles(self, n=128, batch_size=128, temperature=1.):
        batch_sizes = [batch_size for _ in range(n // batch_size)] + [n % batch_size]
        smiles_sampled = []
        likelihoods_sampled = []

        logging.debug("Sampling {} SMILES from Model".format(n))

        for size in batch_sizes:
            if not size:
                break
            # Sample from Agent
            seqs, likelihoods, _ = self.sample(size, temperature=temperature)
            smiles = self.sequence_to_smiles(seqs)
            smiles_sampled.extend(smiles)

            likelihoods_sampled.extend(likelihoods.data.cpu().numpy().tolist())
            # We delete these manually such Pytorch can reuse our precious gpu memory for the next sample
            del seqs, likelihoods, _

        return (smiles_sampled, likelihoods_sampled)
