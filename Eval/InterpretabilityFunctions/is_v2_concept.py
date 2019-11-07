import numpy as np
from sklearn.preprocessing import normalize
from Utils.Loaders.embedding import Embedding
from Utils.Loaders.semcat import SemCat
import tqdm
import multiprocessing


def V_p(V, embedding, n_j, lamb):
    return set([embedding.i2w[int(o)] for o in V[1, -lamb * n_j:]])

def V_n(V, embedding, n_j, lamb):
    return set([embedding.i2w[int(o)] for o in V[1, :lamb * n_j]])

def is_p(i, j, es: np.ndarray, embedding: Embedding, semcat: SemCat, lamb, validation):
    V_1 = np.array([es[:, j]])
    V_2 = np.array([np.arange(V_1.shape[1])])
    V = np.append(V_1, V_2, axis=0)
    V_sorted = V[:, V[0, :].argsort()]

    if validation:
        S = set(semcat.dropped_words[semcat.i2c[i]])
        n_i = semcat.dropped_words[semcat.i2c[i]].__len__()
    else:
        S = set(semcat.vocab[semcat.i2c[i]])
        n_i = semcat.vocab[semcat.i2c[i]].__len__()

    v_p = V_p(V_sorted, embedding, n_i, lamb)
    v_n = V_n(V_sorted, embedding, n_i, lamb)

    IS_p = S.intersection(v_p).__len__() / n_i * 100
    IS_n = S.intersection(v_n).__len__() / n_i * 100

    return max(IS_p, IS_n)


def j_star(i: int, distance_matrix: np.ndarray):
    return int(np.argmax(distance_matrix[:,i]).astype(dtype=np.int))


def is_i(i: int, es: np.ndarray, embedding: Embedding, semcat: SemCat, distance_matrix: np.ndarray, l, validation):
    IS_ji = []
    D = embedding.W.shape[1]
    for j in range(D):
        IS_ji.append(is_p(i, j, es, embedding, semcat, l, validation))

    return IS_ji[j_star(i, distance_matrix)]

def score_dist(embedding_space: np.ndarray, embedding: Embedding, semcat: SemCat, distance_space: np.ndarray, seq, valiudation, lamb=5):
    IS_i = []
    for i in seq:
        IS_i.append(is_i(i, embedding_space, embedding,
                         semcat, distance_space,
                         lamb, valiudation))
    return IS_i


def score(embedding_space: np.ndarray, embedding: Embedding, semcat: SemCat, distance_space: np.ndarray, lamb=5, norm=False, avg=False, validation=False):
    IS_i = []
    C = distance_space.shape[1]

    pool = multiprocessing.Pool(processes=4)

    if norm:
       embedding_space = normalize(embedding_space)

    inputs = []
    for i in range(4):
        seq = [k for k in range(int(C / 4 * i), int(C / 4 * (i + 1)))]
        inputs.append([embedding_space, embedding,
                       semcat, distance_space,
                       seq, validation, lamb])

    with pool as p:
        result = p.starmap(score_dist, inputs)

    for res in result:
        IS_i += res

    if avg:
        return sum(IS_i)/C
    else:
        return IS_i
