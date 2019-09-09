import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


def plot_interpretability_scores(input_file, output_file=None):

    scores = []
    with open(input_file, mode="r", encoding="utf8") as f:
        for l in f.readlines():
            scores.append(float(l))

    lamb = np.arange(len(scores))

    fig, ax = plt.subplots()
    ax.plot(lamb, scores)

    ax.set(xlabel='Lambda', ylabel='Score (%)',
           title='Interpretability Score')
    ax.set_xticks(lamb)
    ax.grid()
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Glove interpretibility')

    parser.add_argument("input_file", type=str,
                        help="Input file which contains the list of the scores")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    plot_interpretability_scores(input_file, output_file)