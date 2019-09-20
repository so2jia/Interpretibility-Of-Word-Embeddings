import os
import sys
import tqdm

import numpy as np
from Utils.Loaders.embedding import Embedding
from Utils.Loaders.semcat import SemCat
from sklearn.neighbors import KernelDensity

from scipy.integrate import quad
import seaborn

import matplotlib.pyplot as plt

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

    if type(p) == float:
        p = np.array([p])

    if type(q) == float:
        q = np.array([q])

    # smp = np.linspace(-1.5, 1.2, 1000000)[:, np.newaxis]

    # plt.plot(smp[:, 0], r)
    # plt.plot(smp[:, 0], e_p)
    # plt.plot(smp[:, 0], e_q)
    # # plt.plot(p, np.arange(p.shape[0]))
    # # seaborn.kdeplot(p)
    # plt.show()

    p_kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    q_kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    _p = p[:, np.newaxis]
    p_kde_fit = p_kde.fit(_p)

    _q = q[:, np.newaxis]
    q_kde_fit = q_kde.fit(_q)

    def g(x, __p, __q):
        p_kde_score = __p.score_samples(np.array([[x]]))

        q_kde_score = __q.score_samples(np.array([[x]]))

        e_p = np.exp(p_kde_score)
        e_q = np.exp(q_kde_score)

        r = np.sqrt(e_p * e_q)
        return r

    ig = quad(g, -np.inf, np.inf, args=(p_kde_fit, q_kde_fit))

    return -np.log1p(ig[0])


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

    for i in tqdm.tqdm(dimension_indexes):
        for j in tqdm.trange(category_size):
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

            b = bhatta_distance(_p, _q)

            # distance
            W_b[i][j] = b
    logging.info(f"Bhattacharya matrix: slice #{id}/{max_id} calculated...")
    return W_b, None


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
        if slice[1] is not None:
            W_bs += np.array(slice[1])

    # Saving matrix
    if save_weights:
        prefix = os.path.join(os.getcwd(), weights_dir)
        dest = os.path.join(prefix, 'w_b.npy')
        np.save(dest, W_b)
        np.save(os.path.join(prefix, 'w_bs.npy'), W_bs)

    return W_b, W_bs