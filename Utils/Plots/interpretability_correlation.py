import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.stats as stats
from argparse import ArgumentParser
from Utils.Loaders.semcat import read as semcat_reader, SemCat
from Utils.Loaders.embedding import read as embedding_reader, Embedding

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


def interpretability_score(e_s: np.ndarray, w_b: np.ndarray, embedding: Embedding, semcat: SemCat, lamb=5):
    logging.info("Calculating interpretability...")

    IS_i = []

    D = e_s.shape[1]
    K = semcat.i2c.__len__()

    for d in range(D):
        D_1 = np.array([e_s[:, d]])
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
        for j in range(IS_ij.__len__()):
            T_1 = np.array([w_b[j, :]])
            T_2 = np.array([np.arange(T_1.shape[1])])
            wb_j = np.append(T_1, T_2, axis=0)
            wb_j_sorted = wb_j[:, wb_j[0, :].argsort()]
            max_j = wb_j_sorted[1, -1]

        IS_i.append(IS_ij[int(max_j)])

    return np.array(IS_i)


def test(input_file: str, bhatta:str, test_type:str, embedding_path, semcat_dir, output_file=None):
    """
    Makes a scatter plot of the dimensional interpretability score according to the KS-test value
    Parameters
    ----------
    input_file: str
        Path to Input file
    output_file: str
        Path to save plot as PNG (Optional)

    Returns
    -------

    """

    embedding = embedding_reader(embedding_path, True, 50000)
    semcat = semcat_reader(semcat_dir)

    test_types = {
        "ks": ks_test,
        "normal": normal_test,
        "shapiro": shapiro,
        "anderson": None,
    }

    e_s = np.load(input_file)
    w_b = np.load(bhatta)

    val, p, name = test_types[test_type](e_s)
    score = interpretability_score(e_s, w_b, embedding, semcat)

    correlate = stats.pearsonr(score, p)

    print(f"Pearson r: {correlate}")

    plotting(score, val, p, name, output_file)


def plotting(dim, ks, p, name, output):
    #TODO fix title overlap
    #TODO add correlation coefficience on plot

    fig: Figure
    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].scatter(ks, dim)
    axs[0].set_xlabel("KS-test value")
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
                        help="Test types: [ks, normal, shapiro, anderson]")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    embedding = args.embedding
    embedding_s = args.embedding_s
    bhatta = args.bhatta
    semcat_dir = args.semcat_dir
    test_type = args.test_type
    output_file = args.output_file

    test(input_file=embedding_s, bhatta=bhatta, test_type=test_type, embedding_path=embedding, semcat_dir=semcat_dir,
         output_file=output_file)
