import numpy as np
from sklearn.preprocessing import normalize
from Utils.Loaders.embedding import Embedding
from Utils.Loaders.semcat import SemCat


def V_p(V, embedding, n_j, lamb):
    return set([embedding.i2w[int(o)] for o in V[1, -lamb * n_j:]])


def V_n(V, embedding, n_j, lamb):
    return set([embedding.i2w[int(o)] for o in V[1, :lamb * n_j]])


def is_ij(j, V_sorted, embedding, semcat, lamb):

    S = set(semcat.vocab[semcat.i2c[j]])
    n_j = semcat.vocab[semcat.i2c[j]].__len__()

    v_p = V_p(V_sorted, embedding, n_j, lamb)
    v_n = V_n(V_sorted, embedding, n_j, lamb)

    IS_p = S.intersection(v_p).__len__() / n_j * 100
    IS_n = S.intersection(v_n).__len__() / n_j * 100

    # The max of positive and negative direction
    IS_b = max(IS_p, IS_n)
    return IS_b


def is_ij_(i: int, distance_space: np.ndarray, embedding_space: np.ndarray, embedding: Embedding, semcat: SemCat, lamb=10):
    K = distance_space.shape[1]

    V_1 = np.array([embedding_space[:, i]])
    V_2 = np.array([np.arange(V_1.shape[1])])
    V = np.append(V_1, V_2, axis=0)
    V_sorted = V[:, V[0, :].argsort()]

    IS_ij = []
    for j in range(K):
        IS_ij.append(is_ij(j, V_sorted, embedding, semcat, lamb))

    max_j = np.argmax(distance_space[i, :])
    return IS_ij[int(max_j)]


def dimensional_score(embedding_space: np.ndarray, embedding: Embedding, semcat: SemCat, distance_space: np.ndarray, lamb=5, norm=False):
    IS_i = []
    D = distance_space.shape[0]
    if norm:
        embedding_space = normalize(embedding_space)
    for i in range(D):

        IS_i.append(is_ij_(i, distance_space, embedding_space, embedding, semcat, lamb=lamb))
    return np.array(IS_i)

def score(embedding_space: np.ndarray, embedding: Embedding, semcat: SemCat, distance_space: np.ndarray, lamb=5, norm=False):
    IS_i = dimensional_score(embedding_space, embedding, semcat, distance_space, lamb, norm)
    return sum(IS_i)/IS_i.shape[0]