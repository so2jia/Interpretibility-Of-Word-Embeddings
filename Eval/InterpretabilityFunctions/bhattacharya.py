import os
import sys

import numpy as np
from Utils.Loaders.embedding import Embedding
from Utils.Loaders.semcat import SemCat

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def bhatta_distance(p: np.ndarray, q: np.ndarray):
    """
    Calculates Bhattacharya distance between two vectors
    Parameters
    ----------
    p
        The category sampled vector
    q
        The out of category sampled vector
    Returns
    -------
    tuple
        Returns with the distance and the sign of the difference of means
    """
    # Variance of p and q
    var1 = np.std(p) ** 2
    var2 = np.std(q) ** 2

    # Mean of p and q
    mean1 = np.mean(p)
    mean2 = np.mean(q)

    # Formula
    x = np.log1p((var1 / var2 + var2 / var1 + 2) / 4) / 4
    bc = x + ((mean1 - mean2) ** 2 / (var1 + var2)) / 4
    sign = -1 if mean1 - mean2 < 0 else 1
    return bc, sign


def calculation_process(embedding: Embedding, semcat: SemCat, category_size: int,
                        dimension_indexes: list, id: int, max_id: int):
    """
    Calculates a slice of the Bhattacharya distance matrix
    Parameters
    ----------
    embedding
        The Embedding object
    semcat
        The SemCat object
    category_size
        The number of categories
    dimension_indexes
        A list of dimension indexes
    id
        The number of the slice (logging)
    max_id
        The number of the slices (logging)

    Returns
    -------

    """
    W_b = np.zeros([embedding.W.shape[1], category_size], dtype=np.float)
    W_bs = np.zeros([embedding.W.shape[1], category_size], dtype=np.int)

    for i in dimension_indexes:
        for j in range(category_size):
            word_indexes = np.zeros(shape=[embedding.W.shape[0], ], dtype=np.bool)
            _p = []
            _q = []
            # Populate P with category word weights
            for word in semcat.vocab[semcat.i2c[j]]:
                try:
                    word_indexes[embedding.w2i[word]] = True
                except KeyError:
                    continue
            _p = embedding.W[word_indexes, i]
            # Populate Q with out of category word weights
            _q = embedding.W[~word_indexes, i]
            # calculating distance
            b, s = bhatta_distance(_p, _q)

            # distance
            W_b[i][j] = b
            # sign
            W_bs[i][j] = s
    logging.info(f"Bhattacharya matrix: slice #{id}/{max_id} calculated...")
    return W_b, W_bs


def bhattacharya_matrix(embedding: Embedding, semcat: SemCat, weights_dir="out", save_weights=False, load_weights=True):
    """
    Calculating Bhattacharya distance matrix
    Parameters
    ----------
    embedding
        Embedding object
    semcat
        SemCat object
    weights_dir
        The path of a directory where the weights will be saved or loaded from
    save_weights
        Save weights
    load_weights
        Load weight
    Returns
    -------
    tuple
        Disarnce Matrix and sign matrix
    """
    # W_b Matrix
    W_b = np.zeros([embedding.W.shape[1], semcat.vocab.__len__()], dtype=np.float)
    W_bs = np.zeros(W_b.shape, dtype=np.int)

    # Loading Bhattacharya matrix
    if load_weights:
        prefix = os.path.join(os.getcwd(), weights_dir)
        logging.info("Loading Bhattacharya distance matrix...")
        if not os.path.exists(prefix):
            logging.info(f"Directory does not exists: {prefix}")
            sys.exit(1)

        # Distance matrix
        W_b = np.load(os.path.join(prefix, '/w_b.npy'))
        # Distance matrix signs
        W_bs = np.load(os.path.join(prefix, '/w_bs.npy'))

        logging.info("Bhattacharya distance matrix loaded!")
        return W_b, W_bs

    # Calculating distance matrix
    number_of_slices = 4

    d = W_b.shape[0]

    logging.info(f"Calculating Bhattacharya distance with {number_of_slices} slices!")

    indexes = [[i for i in range(int(d/number_of_slices*p), int(d/number_of_slices*(p+1)))] for p in range(number_of_slices)]

    slices = [calculation_process(embedding, semcat, W_b.shape[1], i, k+1, len(indexes)) for k, i in enumerate(indexes)]

    for slice in slices:
        W_b += np.array(slice[0])
        W_bs += np.array(slice[1])

    # Saving matrix
    if save_weights:
        prefix = os.path.join(os.getcwd(), weights_dir)
        dest = os.path.join(prefix, 'w_b.npy')
        np.save(dest, W_b)
        np.save(os.path.join(prefix, 'w_bs.npy'), W_bs)

    return W_b, W_bs
