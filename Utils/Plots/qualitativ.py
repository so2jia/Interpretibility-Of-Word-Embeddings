from argparse import ArgumentParser
import numpy as np
from Utils.Loaders.semcat import read as semcat_reader, SemCat
from Utils.Loaders.embedding import read as embedding_reader, Embedding
from sklearn.preprocessing import normalize


def normality(embedding: Embedding, semcat: SemCat, w_b: np.ndarray, lamb=5):

    e_n = normalize(embedding.W)

    IS_i = []

    D = embedding.W.shape[1]
    K = semcat.i2c.__len__()

    for d in range(D):
        D_1 = np.array([e_n[:, d]])
        D_2 = np.array([np.arange(D_1.shape[1])])
        di = np.append(D_1, D_2, axis=0)
        di_sorted = di[:, di[0, :].argsort()]

        IS_ij = []

        for j in range(K):
            S = set(semcat.vocab[semcat.i2c[j]])
            n_j = semcat.vocab[semcat.i2c[j]].__len__()

            V_p = set([embedding.i2w[int(o)] for o in di_sorted[1, -lamb * n_j:]])
            V_n = set([embedding.i2w[int(o)] for o in di_sorted[1, :lamb * n_j]])

            IS_p = S.intersection(V_p).__len__() / n_j * 100
            IS_n = S.intersection(V_n).__len__() / n_j * 100

            # The max of positive and negative direction
            IS_b = max(IS_p, IS_n)

            IS_ij.append(IS_b)

        max_j = 0
        max_j_val = -np.inf
        for j in range(IS_ij.__len__()):
            T_1 = np.array([w_b[j, :]])
            T_2 = np.array([np.arange(T_1.shape[1])])
            wb_j = np.append(T_1, T_2, axis=0)
            wb_j_sorted = wb_j[:, wb_j[0, :].argsort()]
            if max_j_val < wb_j_sorted[0, -1]:
                max_j = wb_j_sorted[1, -1]
                max_j_val = wb_j_sorted[0, -1]

        IS_i.append(IS_ij[int(max_j)])

    return np.array(IS_i)


def q(bhatta:str, embedding_path, semcat_dir, output_file=None):
    embedding = embedding_reader(embedding_path, True, 50000)
    semcat = semcat_reader(semcat_dir)

    w_b = np.load(bhatta)

    normality(embedding, semcat, w_b)


if __name__ == '__main__':
    parser = ArgumentParser(description='')

    # Required parameters
    parser.add_argument("embedding", type=str,
                        help="Embedding matrix")
    parser.add_argument("bhatta", type=str,
                        help="Bhattacharyya matrix")
    parser.add_argument("semcat_dir", type=str,
                        help="SemCat Categories dir")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    embedding = args.embedding
    bhatta = args.bhatta
    semcat_dir = args.semcat_dir
    output_file = args.output_file

    q(bhatta=bhatta, embedding_path=embedding, semcat_dir=semcat_dir,
      output_file=output_file)
