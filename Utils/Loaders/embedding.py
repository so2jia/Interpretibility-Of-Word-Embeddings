import re
import sys

import argparse
import collections

import numpy as np
import scipy.sparse as sp

import gzip
from zipfile import ZipFile

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class Embedding(object):
    """
    This class provides utils for efficient storage and manipulation of sparse (embedding) matrices.
    Objects are assumed to be located in the rows.
    """
    
    def __init__(self, embedding_path, dense_input, words_to_keep=None, max_words=-1):
        """
        Parameters
        ----------
        embedding_path : str
            Location of the embedding
        dense_input : bool
            Marks if it is a dense embedding
        words_to_keep : list, optional
            list of words to keep
        max_words : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        """
        if dense_input:
            self.w2i, self.i2w, self.W = self.load_dense_embeddings(embedding_path, words_to_keep=words_to_keep, max_words=max_words)
        else:
            self.w2i, self.i2w, self.W = self.load_sparse_embeddings(embedding_path, words_to_keep=words_to_keep, max_words=max_words)

    @staticmethod
    def load_dense_embeddings(path: str, words_to_keep=None, max_words=-1):
        """
        Reads in the dense embedding file.

        Parameters
        ----------
        path : str
            Location of the gzipped dense embedding file
            If None, no filtering takes place.
        words_to_keep : list, optional
            list of words to keep
        max_words : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        Returns
        -------
        tuple:
            w2i:
                Wordform to identifier dictionary,
            i2w:
                Identifier to wordform dictionary,
            W:
                The dense embedding matrix
        """
        if path.endswith('.gz'):
            lines = gzip.open(path, 'rt')
        elif path.endswith('.zip'):
            myzip = ZipFile(path) # we assume only one embedding file to be included in a zip file
            lines = myzip.open(myzip.namelist()[0])
        else:
            lines = open(path)
        data, words = [], []
        for counter, line in enumerate(lines):
            if len(words) % 5000 == 0:
                logging.info("{} lines read in from a dense embedding file".format(len(words)))

            if len(words) == max_words:
                break
            tokens = line.rstrip().split(' ')
            if len(words) == 0 and len(tokens) == 2 and re.match('[1-9][0-9]*', tokens[0]):
                # the first line might contain the number of embeddings and dimensionality of the vectors
                continue
            if words_to_keep is not None and not tokens[0] in words_to_keep:
                continue
            try:
                values = [float(i) for i in tokens[1:]]
                if sum([v**2 for v in values])  > 0: # only embeddings with non-zero norm are kept
                    data.append(values)
                    words.append(tokens[0])
            except:
                print('Error while parsing input line #{}: {}'.format(counter, line))

        i2w = dict(enumerate(words))
        w2i = {v: k for k, v in i2w.items()}
        return w2i, i2w, np.array(data)

    @staticmethod
    def load_sparse_embeddings(path, words_to_keep=None, max_words=-1):
        """
        Reads in the sparse embedding file.

        Parameters
        ----------
        path : str
            Location of the gzipped sparse embedding file
            If None, no filtering takes place.
        words_to_keep : list, optional
            list of words to keep
        max_words : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        Returns
        -------
        tuple:
            w2i:
                Wordform to identifier dictionary,
            i2w:
                Identifier to wordform dictionary,
            W:
                The sparse embedding matrix
        """

        i2w = {}
        data, indices, indptr = [], [], [0]
        with gzip.open(path, 'rt') as f:
            for line_number, line in enumerate(f):

                if len(i2w) % 5000 == 0:
                    logging.info("{} lines read in from a sparse embedding file".format(len(i2w)))

                if line_number == max_words:
                    break
                parts = line.rstrip().split(' ')

                if words_to_keep is not None and parts[0] not in words_to_keep:
                    continue

                i2w[len(i2w)] = parts[0]
                for i, value in enumerate(parts[1:]):
                    value = float(value)
                    if value != 0:
                        data.append(float(value))
                        indices.append(i)
                indptr.append(len(indices))
        w2i = {w:i for i,w in i2w.items()}
        return w2i, i2w, sp.csr_matrix((data, indices, indptr), shape=(len(indptr)-1, i+1))

    def query_by_index(self, idx, top_words=25000, top_k=10):
        assert type(self.W) == sp.csr_matrix  # this method only works for sparse matrices at the moment
        relative_scores = []
        word_ids = []
        for wid, we in enumerate(self.W):
            if wid == top_words:
                break
            if idx in we.indices:
                s = np.sum(we.data)
                for i, d in zip(we.indices, we.data):
                    if i == idx:
                        relative_scores.append(d / s)
                        word_ids.append(wid)
                        break
        order = np.argsort(relative_scores)
        if top_k > 0: order = order[-top_k:]
        return [(self.i2w[word_ids[j]], relative_scores[j], word_ids[j]) for j in order]


def _constructing_mcrae_features(mcrae_dir: str):
    """
    Creating feature dictionary

    Parameters
    ----------
    mcrae_dir: str
        Path to the McRae file

    Returns
    -------
    dict:
        Concept-Index dictionary
    """
    c2i = {}

    # Iterating over the lines of the file
    for i, l in enumerate(open('{}/McRae-BRM-InPress/CONCS_FEATS_concstats_brm.txt'.format(mcrae_dir))):
        # Skipping column labels
        if i == 0:
            continue
        # Split it into columns
        parts = l.split()
        # Creating Concept-Index pairs
        if parts[0] not in c2i:
            c2i[parts[0]] = len(c2i)
    return c2i


def read(input_file: str, dense_file: bool, lines_to_read=-1, mcrae_dir=None, mcrae_words_only=None):
    """
    Reads embedding file
    Parameters
    ----------
        input_file : str
            Path to the input file.
        dense_file : bool
            True if it marks a dense embedding, false otherwise.
        lines_to_read : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        mcrae_dir : str, optional
            Path to the McRae file
        mcrae_words_only : bool
            Use McRae words only
    Returns
    -------
    Embedding:
        The Embedding object
    """
    path_to_embedding = input_file

    c2i = {}
    if mcrae_dir:
        c2i = _constructing_mcrae_features(mcrae_dir)

    if mcrae_dir is not None and mcrae_words_only is not None:
        emb = Embedding(path_to_embedding, dense_file, max_words=lines_to_read, words_to_keep=c2i.keys())
    else:
        emb = Embedding(path_to_embedding, dense_file, max_words=lines_to_read, words_to_keep=None)

    return emb


def main():
    """
    For CLI
    """
    parser = argparse.ArgumentParser(description='Util to read in an embedding file.')
    parser.add_argument('input_file', type=str, help='path to the input file')

    parser.add_argument('--dense_file', dest='dense', action='store_true')
    parser.set_defaults(dense=False)

    parser.add_argument('--lines_to_read', type=int, help='number of embeddings to read', default=-1)

    parser.add_argument('--mcrae_dir', type=str, help='path to the McRae file', default=None)
    parser.add_argument('-mcrae_words_only', action='store_true')
    
    args = parser.parse_args()
    path_to_embedding = args.input_file

    emb = read(path_to_embedding, args.dense, args.lines_to_read, args.mcrae_dir, args.mcrae_words_only)

    logging.info("The embedding matrix read in has a shape of {}".format(emb.W.shape))

    if type(emb.W) == sp.csr_matrix:
        dim_id = 22
        logging.info("top words for dimension {} are".format(dim_id))
        for i, top in enumerate(emb.query_by_index(dim_id)):
            logging.info("#{}: {}".format(i, top))


if __name__ == "__main__":
    main()
