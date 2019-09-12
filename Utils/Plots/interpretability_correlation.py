import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from argparse import ArgumentParser
import scipy.signal as signal


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


def interpretability_score(x: np.ndarray):
    IS = []
    for i in range(x.shape[0]):
        IS.append(np.max(x[i]))
    return np.array(IS)


def test(input_file: str, bhatta:str, test_type:str, output_file=None):
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

    test_types = {
        "ks": ks_test,
        "normal": normal_test,
        "shapiro": shapiro,
        "anderson": None,
    }

    e_s = np.load(input_file)
    w_b = np.load(bhatta)

    val, p, name = test_types[test_type](e_s)
    score = interpretability_score(w_b)

    correlate = stats.pearsonr(score, p)

    plotting(score, val, p, name, output_file)


def plotting(dim, ks, p, name, output):
    #TODO fix title overlap
    #TODO add correlation coefficience on plot

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(name)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)


    ax1.scatter(ks, dim)
    ax1.set_xlabel("KS-test value")
    ax1.set_ylabel("Dimensional interpretability")

    ax2.scatter(p, dim)
    ax2.plot([0.05, 0.05], [dim.min(), dim.max()], color='r', linestyle='-', linewidth=2)
    ax2.set_ylim((dim.min(), dim.max()))
    ax2.set_xlabel("P")
    ax2.set_ylabel("Dimensional interpretability")

    if output_file is not None:
        fig.savefig(output_file)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Normality check and interpretability correlation')

    # Required parameters
    parser.add_argument("embedding_s", type=str,
                        help="Standardised embedding matrix")
    parser.add_argument("bhatta", type=str,
                        help="Bhattacharyya matrix")
    parser.add_argument("test_type", type=str,
                        help="Test types: [ks, normal, shapiro, anderson]")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    embedding_s = args.embedding_s
    bhatta = args.bhatta
    test_type = args.test_type
    output_file = args.output_file

    test(input_file=embedding_s, bhatta=bhatta, test_type=test_type, output_file=output_file)