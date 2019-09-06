import os
import sys

import numpy as np
from Utils.Loaders.embedding import Embedding
from Utils.Loaders.semcat import SemCat

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def bhatta_distance(p, q):
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


def calculation_process(embedding, semcat, category_size, dimension_indexes, id, max_id):
    
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


def bhattacharya_matrix(embedding: Embedding, semcat: SemCat, output_dir="out", save=False, load=True):
    """
    Calculating Bhattacharya distance matrix
    Parameters
    ----------
    embedding
        Embedding object
    semcat
        SemCat object
    output_dir
        The path of a directory where the weights will be saved or loaded from
    save
        Save weights
    load
        Load weight
    Returns
    -------
    tuple
        Disarnce Matrix and sign matrix
    """
    # W_b Matrix
    W_b = np.zeros([embedding.W.shape[1], semcat.vocab.__len__()], dtype=np.float)
    W_bs = np.zeros(W_b.shape, dtype=np.int)

    if load:
        prefix = os.path.join(os.getcwd(), output_dir)
        logging.info("Loading Bhattacharya distance matrix...")
        if not os.path.exists(prefix):
            logging.info(f"Directory does not exists: {prefix}")
            sys.exit(1)

        # Distance matrix
        W_b = np.load(os.path.join(prefix, '/wb.npy'))
        # Distance matrix signs
        W_bs = np.load(os.path.join(prefix, '/wbs.npy'))

        logging.info("Bhattacharya distance matrix loaded!")
        return W_b, W_bs


    number_of_processes = 4

    d = W_b.shape[0]

    logging.info(f"Calculating Bhattacharya distance with {number_of_processes} processes!")

    indexes = [[i for i in range(int(d/number_of_processes*p), int(d/number_of_processes*(p+1)))] for p in range(number_of_processes)]

    slices = [calculation_process(embedding, semcat, W_b.shape[1], i, k+1, len(indexes)) for k, i in enumerate(indexes)]

    for slice in slices:
        W_b += np.array(slice[0])
        W_bs += np.array(slice[1])


    # if i % 10 == 0:
    #     logging.info(f"Calculating W_b... ({i + 1}/{W_b.shape[0]})")

    if save:
        prefix = os.path.join(os.getcwd(), output_dir)
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        dest = os.path.join(prefix, 'wb.npy')
        np.save(dest, W_b)
        np.save(os.path.join(prefix, 'wbs.npy'), W_bs)

    return W_b, W_bs
