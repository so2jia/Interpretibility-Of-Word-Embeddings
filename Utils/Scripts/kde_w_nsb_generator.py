import os
import tqdm

import numpy as np
from Utils.Loaders.embedding import Embedding, read as embreader
from Utils.Loaders.semcat import SemCat, read as smreader

import multiprocessing

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

    # Mean of p and q
    mean1 = np.mean(p)
    mean2 = np.mean(q)

    if (mean1 - mean2) < 0:
        return -1
    return 1


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

    # TODO reduce the number of progress bars

    for k in tqdm.trange(dimension_indexes.__len__(), unit='dim', desc=f'__ On PID - {os.getpid()}\t'):
        i = dimension_indexes[k]
        for j in tqdm.trange(category_size, desc=f'>> On PID - {os.getpid()}\t'):
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

            # sign
            W_bs[i][j] = b

    logging.info(f"Bhattacharya matrix: slice #{id}/{max_id} calculated...")
    return None, W_bs


def bhattacharya_matrix(embedding: Embedding, semcat: SemCat,
                        weights_dir="out", processes=2, name="default"):
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

    # Calculating distance matrix
    number_of_slices = processes

    d = W_b.shape[0]

    logging.info(f"Calculating Bhattacharya distance with {number_of_slices} slices!")

    indexes = [[i for i in range(int(d/number_of_slices*p), int(d/number_of_slices*(p+1)))] for p in range(number_of_slices)]

    slices = [
        [embedding,
         semcat,
         W_b.shape[1],
         i,
         k+1,
         len(indexes)]
        for k, i in enumerate(indexes)
    ]

    with multiprocessing.Pool(number_of_slices) as pool:
        result = pool.starmap(calculation_process, slices)

    for slc in result:
        W_bs += np.array(slc[1])

    # Saving matrix
    prefix = os.path.join(os.getcwd(), weights_dir)
    np.save(os.path.join(prefix, f'{name}_w_bs.npy'), W_bs)


if __name__ == '__main__':
    regul = "5"
    emb = embreader(f"../../data/glove/glove.6B.300d.txt", True, 50000)
    sm = smreader("../../data/semcat/Categories/")

    bhattacharya_matrix(emb, sm, "../../out/kde/dense_model/", name=f"raw")

    w_nb = np.load(f"../../out/kde/dense_model/raw_w_nb.npy")
    w_sb = np.load(f"../../out/kde/dense_model/raw_w_bs.npy")

    w_nsb = w_nb * w_sb

    np.save(f"../../out/kde/dense_model/raw_w_nsb.npy", w_nsb)


    # sparse
    # regul = "5"
    # emb = embreader(f"../../data/glove/sparse/glove300d_l_0.{regul}_DL_top50000.emb.gz", False, 50000)
    # sm = smreader("../../data/semcat/Categories/")
    #
    # bhattacharya_matrix(emb, sm, "../../out/", name=f"l_0.{regul}")
    #
    # w_nb = np.load(f"../../out/kde/sparse_model/l_0.{regul}_w_nb.npy")
    # w_sb = np.load(f"../../out/kde/sparse_model/l_0.{regul}_w_bs.npy")
    #
    # w_nsb = w_nb*w_sb
    #
    # np.save(f"../../out/kde/sparse_model/l_0.{regul}_w_nsb.npy", w_nsb)