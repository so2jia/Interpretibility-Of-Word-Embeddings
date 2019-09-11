import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from argparse import ArgumentParser
from Utils.Loaders.semcat import read as semcat_reader, SemCat
from Utils.Loaders.embedding import read as embedding_reader, Embedding


def interpretability_score_of_dimensions(embedding: Embedding, I: np.ndarray, semcat: SemCat, lamb=1) -> np.ndarray:
    """
    Calculates dimensional interpretability scores
    Parameters
    ----------
    embedding: Embedding
        Embedding object
    I: array-like
        Category-Embedding weights
    semcat: SemCat
        SemCat Object
    lamb: int
        Lambda value
    Returns
    -------
    np:
        Dimensional scores
    """
    d = I.shape[1]
    k = semcat.i2c.__len__()

    IS_d = []

    for i in range(d):
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
        # Interpretability csore of dimensions
        IS_d.append(IS_m)
    return np.array(IS_d)

def plot_ks_test_dim_interpret(input_file:str, embedding_path:str, semcat_path:str, lines_to_read:int, output_file= None):
    """
    Makes a scatter plot of the dimensional interpretability score according to the KS-test value
    Parameters
    ----------
    input_file: str
        Path toCategory-Embedding matrix (I.npy)
    embedding_path: str
        Path to Embedding file
    semcat_path: str
        Path to SemCat categories directory
    lines_to_read: int
        Maximum lines to read from embedding
    output_file: str
        Path to save plot as PNG (Optional)

    Returns
    -------

    """
    I = np.load(input_file)
    embedding = embedding_reader(embedding_path, dense_file=True,
                                 lines_to_read=lines_to_read if lines_to_read is not None else -1)
    semcat = semcat_reader(semcat_path)

    dim_interpretability = interpretability_score_of_dimensions(embedding, I, semcat, 5)
    ks_test = np.zeros(dim_interpretability.shape[0])
    p_values = np.zeros(dim_interpretability.shape[0])

    for i in range(dim_interpretability.shape[0]):
        d, p = stats.kstest(I[:, i], 'norm')
        ks_test[i] = d
        p_values[i] = p

    ks_test = np.array(ks_test)

    ax_scatter = plt.scatter(ks_test, dim_interpretability)
    plt.xlabel("KS-test value")
    plt.ylabel("Dimensional interpretability")

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='KS-test, Dimensional Interpretability scatterplot')

    # Required parameters
    parser.add_argument("input_file", type=str,
                        help="Category-Embedding matrix (I.npy)")
    parser.add_argument("embedding_file", type=str,
                        help="Embedding file path. It has to be dense and the same as in the evaluation process.")
    parser.add_argument("semcat_dir", type=str,
                        help="SemCat categories directory")
    parser.add_argument("--lines_to_read", type=int,
                        help="Max lines to read. it has to be the same amount as in the evaluation. Default -1 (Optinal)")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    input_file = args.input_file
    embedding_file = args.embedding_file
    semcat_dir = args.semcat_dir
    lines_to_read = args.lines_to_read
    output_file = args.output_file

    plot_ks_test_dim_interpret(input_file=input_file, embedding_path=embedding_file, semcat_path=semcat_dir,
                               lines_to_read=lines_to_read, output_file=output_file)