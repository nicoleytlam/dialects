"""
Code for loading data and abstractions for representing data. DO NOT
EDIT THIS FILE.
"""
from typing import Generator, List, Tuple

import numpy as np
import torch
import re
import random

from _utils import cache_pickle, timer


class Vocabulary(object):
    """
    A container that maintains a mapping between tokens and
    their indices.
    """

    def __init__(self, forms: List[str]):
        self.forms = forms
        self.indices = {j: i for i, j in enumerate(forms)}

    def get_index(self, form: str) -> int:
        """
        Looks up the index of a token or POS tag.

        :param form: The string form of the token or POS tag
        :return: The index for the token or POS tag
        """
        return self.indices[form]

    def get_form(self, index: int) -> str:
        """
        Looks up the string form of a token or POS tag.

        :param index: The index for the token or POS tag
        :return: The string form of the token or POS tag
        """
        return self.forms[index]

    def __contains__(self, item: str) -> bool:
        return item in self.forms

    def __len__(self) -> int:
        return len(self.forms)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.forms)
    
    @classmethod
    @cache_pickle
    def from_paired_file(cls, raw_data_filename: str):
        """
        Creates a Vocabulary from a file of strings. This function reads the
        file and includes all of the tokens in the vocabulary, along with [BOS] and [PAD].

        :param raw_data_filename: The name of the file
        :param cache_filename: If this keyword argument is provided, the
            vocabulary will be cached to a pickle file whose filename is
            given by this keyword argument
        :return: A Vocabulary including the tokens in the file
        """

        words = []
        with open(raw_data_filename) as f:
            for line in f:
                line = line.split()
                words += line
        words = ["[BOS]", "[PAD]"] + list(set(words))
        return cls(words)


class Dataset(object):
    """
    A container that stores a dataset and divides it into batches.
    """

    def __init__(self, inputs: List[List[int]], outputs: List[List[int]],
                 pad_index: int, sort_by_length: bool = True):
        """
        Creates a dataset from pre-processed data. Data should be in the
        form of lists of lists of indices.

        :param sentences: The sentences that need to be POS-tagged.
        :param pos_tags: The POS tags corresponding to the sentences.
        :param pad_index: The index of the [PAD] token within the
            token vocabulary.
        :param sort_by_length: If True, the dataset will be sorted by
            length
        """
        self.pad_index = pad_index
        all_data = list(zip(inputs, outputs))
        random.shuffle(all_data)

        # Sort data by length
        if sort_by_length:
            all_data.sort(key=lambda x: len(x[0]))
        self._inputs, self._outputs = zip(*all_data)

    def __len__(self):
        return len(self._inputs)

    @property
    def labels(self):
        return self._outputs

    def get_batches(self, batch_size: int, pad_input_left: bool = True) \
            -> Generator[Tuple[torch.LongTensor, torch.LongTensor], None,
                         None]:
        """
        Creates a generator that loops over the data in the form of
        mini-batches.

        :param batch_size: The size of each mini-batch
        :return: Each item in the generator should contain an array of
            inputs of shape (batch size, max input length) and an array of outputs of shape (batch size, max output length).
        """
        for i in range(0, len(self), batch_size):
            j = i + batch_size
            
            # Pad the batch
            input_len = max(len(s) for s in self._inputs[i:j])
            output_len = max(len(s) for s in self._outputs[i:j])
            if pad_input_left:
                inputs = [[self.pad_index] * (input_len - len(s)) + s 
                          for s in self._inputs[i:j]]
            else:
                inputs = [s + [self.pad_index] * (input_len - len(s)) 
                          for s in self._inputs[i:j]]
            outputs = [o + [self.pad_index] * (output_len - len(o)) 
                       for o in self._outputs[i:j]]

            yield torch.LongTensor(inputs), torch.LongTensor(outputs)

    @classmethod
    @cache_pickle
    def from_paired_file(cls, raw_data_filename: str, vocab: Vocabulary, sort_by_length: bool = True):
        """
        Creates a Dataset from a file of strings. This function reads the
        file and processes the data into index lists.

        :param raw_data_filename: The name of the file
        :param vocab: A Vocabulary for the tokens
        :param cache_filename: If this keyword argument is provided, the
            embeddings will be cached to a pickle file whose filename is
            given by this keyword argument
        :return: A Dataset contatining the data in the file
        """

        inputs = []
        targets = []
        words = []
        with open(raw_data_filename) as f:
            for line in f:
                line = re.split(r'\t+', line)
                input, target = line[0].split(), line[1].split()
                input = [vocab.get_index(w) for w in input]
                target = [vocab.get_index(w) for w in target]
                inputs += [input]
                targets += [target]
                words += input + target
            words = list(set(words))
                
        return cls(inputs, targets, vocab.get_index("[PAD]"), 
                   sort_by_length=sort_by_length)
    
    @classmethod
    def reverse_dataset(cls, vocab: Vocabulary, min_length: int, 
                        max_length: int, size: int):
        non_words = { vocab.get_index('[BOS]'), vocab.get_index('[PAD]')}
        indices = list(set(vocab.indices.values()).difference(non_words))
        inputs = []
        for i in range(size):
            example_length = random.randint(min_length, max_length)
            inputs += [list(random.choices(indices, k=example_length))]
        outputs = [i[-1::-1] for i in inputs]
        return cls(inputs, outputs, pad_index=vocab.get_index('[PAD]'))
    
    @classmethod
    def copy_dataset(cls, vocab: Vocabulary, min_length: int, 
                        max_length: int, size: int):
        non_words = { vocab.get_index('[BOS]'), vocab.get_index('[PAD]')}
        indices = list(set(vocab.indices.values()).difference(non_words))
        inputs = []
        for i in range(size):
            example_length = random.randint(min_length, max_length)
            inputs += [list(random.choices(indices, k=example_length))]
        outputs = inputs
        return cls(inputs, outputs, pad_index=vocab.get_index('[PAD]'))
