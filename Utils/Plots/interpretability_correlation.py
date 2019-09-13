import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.stats as stats
from argparse import ArgumentParser
from Utils.Loaders.semcat import read as semcat_reader, SemCat
from Utils.Loaders.embedding import read as embedding_reader, Embedding
from sklearn.preprocessing import normalize
import os

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def ks_test(x: np.ndarray):
    ks_test = np.zeros(x.shape[1])
    p_values = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        d, p = stats.kstest(x[:, i], 'norm')
        ks_test[i] = d
        p_values[i] = p
    return ks_test, p_values, "KS Test"


def normal_test(x: np.ndarray):
    normal_test = np.zeros(x.shape[1])
    p_values = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        d, p = stats.normaltest(x[:, i])
        normal_test[i] = d
        p_values[i] = p
    return normal_test, p_values, "Normal Test"


def shapiro(x: np.ndarray):
    shapiro = np.zeros(x.shape[1])
    p_values = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        d, p = stats.shapiro(x[:, i])
        shapiro[i] = d
        p_values[i] = p
    return shapiro, p_values, "Shapiro Test"


def anderson(x: np.ndarray):
    anderson = np.zeros(x.shape[1])
    p_values = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        d, cv, p = stats.anderson(x[:, i], 'norm')
        anderson[i] = d
        p_values[i] = p
    return anderson, p_values


def interpretability_score(e_s: np.ndarray, w_b: np.ndarray, embedding: Embedding, semcat: SemCat, lamb=5, norm=False):
    logging.info("Calculating interpretability...")

    IS_i = []

    D = embedding.W.shape[1]
    K = semcat.i2c.__len__()

    embedding_space = embedding.W

    if norm:
        embedding_space = normalize(embedding_space)

    for d in range(D):
        D_1 = np.array([embedding_space[:, d]])
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

        max_j = np.argmax(w_b[d, :])

        IS_i.append(IS_ij[int(max_j)])

    return np.array(IS_i)


def test(input_file: str, bhatta: str, test_type: str, embedding_path: str, semcat_dir: str, params: dict):
    """
    Normality test and interpretability score correlation
    Parameters
    ----------
    input_file: str
        File of standardised embedding matrix
    bhatta: str
        File of Bhattacharyya distance matrix
    test_type: str
        Test types: [ks, normal, shapiro]
    embedding_path: str
        Embedding file
    semcat_dir: str
        SemCat directory
    params: dict
        params = {
            "norm": bool,
            "dense": bool,
            "lines_to_read": int,
            "output_file": str
        }
    Returns
    -------

    """

    embedding = embedding_reader(embedding_path, params["dense"], params["lines_to_read"])
    semcat = semcat_reader(semcat_dir)

    test_types = {
        "ks": ks_test,
        "normal": normal_test,
        "shapiro": shapiro,
    }

    e_s = np.load(input_file)
    w_b = np.load(bhatta)

    val, p, name = test_types[test_type](e_s)
    score = interpretability_score(e_s, w_b, embedding, semcat, norm=params["norm"])

    correlate = stats.pearsonr(score, p)

    path = params["output_file"].split('/')
    fp = path[-1].split('.')[0:-1]
    dir_path = ""
    for x in path[:-1]:
        dir_path.join(x)

    fname = ""
    for x in fp:
        fname.join(x)

    fname.join("-stats.txt")

    p = os.path.join(dir_path, fname)
    with open(p, mode="w", encoding="utf8") as f:
        f.write(f"# Pearson (R, P)\n{correlate}\n# IS\n{sum(score)/score.shape[0]}")

    logging.info(f"Pearson r: {correlate}")
    logging.info(f"IS':{sum(score)/score.shape[0]}")

    plotting(score, val, p, name, params["output_file"])


def plotting(dim, ks, p, name, output):
    #TODO fix title overlap
    #TODO add correlation coefficience on plot

    fig: Figure
    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].scatter(ks, dim)
    axs[0].set_xlabel("Test value")
    axs[0].set_ylabel("Dimensional interpretability")
    fig.suptitle(name, fontsize=16)

    axs[1].scatter(p, dim)
    axs[1].plot([0.05, 0.05], [dim.min(), dim.max()], color='r', linestyle='-', linewidth=2)
    axs[1].set_ylim((dim.min(), dim.max()))
    axs[1].set_xlabel("P")
    axs[1].set_ylabel("Dimensional interpretability")

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    if output is not None:
        fig.savefig(output)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Normality check and interpretability correlation')

    # Required parameters
    parser.add_argument("embedding", type=str,
                        help="Embedding matrix")
    parser.add_argument("embedding_s", type=str,
                        help="Standardised embedding matrix")
    parser.add_argument("bhatta", type=str,
                        help="Bhattacharyya matrix")
    parser.add_argument("semcat_dir", type=str,
                        help="SemCat Categories dir")
    parser.add_argument("test_type", type=str,
                        help="Test types: [ks, normal, shapiro]")
    parser.add_argument("-norm", type=bool,
                        help="l2 norm of embedding space")
    parser.add_argument("-dense", type=bool,
                        help="Dense embedding")
    parser.add_argument("--line_to_read", type=int, default=50000,
                        help="Maximum line to read from embedding file")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    embedding = args.embedding
    embedding_s = args.embedding_s
    bhatta = args.bhatta
    semcat_dir = args.semcat_dir
    test_type = args.test_type
    norm = args.norm
    dense = args.dense
    line_to_read = args.line_to_read
    output_file = args.output_file

    params = {
        "norm": False if norm is None else norm,
        "dense": False if dense is None else dense,
        "lines_to_read": line_to_read,
        "output_file": output_file
    }

    test(input_file=embedding_s, bhatta=bhatta, test_type=test_type, embedding_path=embedding, semcat_dir=semcat_dir,
         params=params)