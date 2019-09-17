import numpy as np
from Utils.Loaders.embedding import Embedding
from Utils.Loaders.semcat import SemCat
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def score(embedding: Embedding, I: np.ndarray, semcat: SemCat, lamb=1):
    """

    Parameters
    ----------
    embedding
        Embedding object
    I
        Category-Embedding weights
    semcat
        SemCat Object
    lamb: int
        Lambda value
    Returns
    -------
    int
        Score (%)
    """
    d = I.shape[1]
    k = semcat.i2c.__len__()

    IS_d = []

    for i in range(d):

        # if i % 5 == 0:
        #     logging.info(f"{i}/{d} dimension done!")

        # Weight-Index matrix
        T_1 = np.array([I[:, i]])
        T_2 = np.array([list(embedding.i2w.keys())])
        iw = np.append(T_1, T_2, axis=0)
        # Sort at the dimension
        iw = iw.T
        iw_sorted = iw[iw[:, 0].argsort()]

        # Max value at a category
        IS_m = -np.inf
        for j in range(k):
            # Words of j-th category
            S = set(semcat.vocab[semcat.i2c[j]])
            # Number of words in the j-th category
            n_j = semcat.vocab[semcat.i2c[j]].__len__()

            # Top and bottom ranking words
            V_p = set([embedding.i2w[int(o)] for o in iw_sorted[-lamb * n_j:, 1]])
            V_n = set([embedding.i2w[int(o)] for o in iw_sorted[:lamb * n_j, 1]])

            IS_p = S.intersection(V_p).__len__() / n_j * 100
            IS_n = S.intersection(V_n).__len__() / n_j * 100

            # The max of positive and negative direction
            IS_b = max(IS_p, IS_n)
            # Getting the max from the category
            if IS_b > IS_m:
                IS_m = IS_b
            # print(f"Dim: {i}, Category: {j}-{semcat.i2c[j]}, IS_p: {IS_p}, IS_n: {IS_n}")
        IS_d.append(IS_m)

    IS = 1 / d * np.sum(IS_d)
    return IS