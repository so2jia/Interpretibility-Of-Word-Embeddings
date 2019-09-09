import matplotlib.pyplot as plt
import numpy as np
from Utils.Loaders.semcat import read
from argparse import ArgumentParser


def plot_semantic_decomposition(semcat: str, input_file:str, output_file=None):
    """
    Creates a plot of semantic decomposition
    Parameters
    ----------
    semcat: str
        Path to SemCat categories directory
    input_file: str
        Path to input file
    output_file: str
        Optional: Path to save figure as PNG

    Returns
    -------

    """
    plt.rcdefaults()
    fig, ax = plt.subplots()

    lines = []
    word = input_file.split('/')[-1].rstrip('.txt').split('-')[-1]
    y_axis_label = []
    x_value = []
    in_cat = []

    semcat = read(semcat)

    with open(input_file, mode="r", encoding="utf8") as f:
        for line in f.readlines():
            l = line.split(' ')
            lines.append(l)
            y_axis_label.append(l[0])
            x_value.append(float(l[2]))
            if word in semcat.vocab[l[0]]:
                in_cat.append(1)
            else:
                in_cat.append(0)

    # Example data
    y_pos = np.arange(len(y_axis_label))

    barlist = ax.barh(y_pos, x_value, align='center')

    for i, v in enumerate(in_cat):
        if v == 1:
            barlist[i].set_color('r')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_axis_label)
    ax.invert_yaxis()
    ax.set_xlabel('')
    ax.set_title(f'Semantical decomposition of word {word}')
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Glove interpretibility')

    # Required parameters
    parser.add_argument("semcat_dir", type=str,
                        help="Path to the SemCat Categories directory")
    parser.add_argument("input_file", type=str,
                        help="Semantic decomposition file")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    semcat_dir = args.semcat_dir
    input_file = args.input_file
    output_file = args.output_file

    plot_semantic_decomposition(semcat_dir, input_file, output_file)
