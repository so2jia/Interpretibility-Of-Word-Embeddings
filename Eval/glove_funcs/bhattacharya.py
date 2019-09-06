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


def bhattacharya_matrix(embedding: Embedding, semcat: SemCat, save=False, load=True):
    """
    Calculating Bhattacharya distance matrix
    Parameters
    ----------
    embedding
        Embedding object
    semcat
        SemCat object
    save
        Save weights
    load
        Load weight
    Returns
    -------
    tuple
        Disarnce Matrix and sign matrix
    """
    epsilon = embedding.W
    # W_b Matrix
    W_b = np.zeros([embedding.W.shape[1], semcat.vocab.__len__()], dtype=np.float)
    W_bs = np.ones(W_b.shape, dtype=np.int)

    if load:
        logging.info("Loading Bhattacharya distance matrix...")
        # Distance matrix
        W_b = np.load('../temp/wb.npy')
        # Distance matrix signs
        W_bs = np.load('../temp/wbs.npy')
        logging.info("Bhattacharya distance matrix loaded!")
        return W_b, W_bs

    # Indexes: i -> dimension, j -> category
    for i in range(W_b.shape[0]):
        for j in range(W_b.shape[1]):
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
        if i % 10 == 0:
            logging.info(f"Calculating W_b... ({i + 1}/{W_b.shape[0]})")
    if save:
        np.save('../temp/wb.npy', W_b)
        np.save('../temp/wbs.npy', W_bs)
    return W_b, W_bs
