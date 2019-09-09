import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


def plot_categorical_decomposition(input_file, dimenstion, output_file=None):
    """
    Plot categorical decomposition
    Parameters
    ----------
    input_file: str
        Path to Category-Weight matrix (w_b.npy)
    dimenstion: int
        The dimension of decomposition
    output_file
        Output PNG file (Optional)
    Returns
    -------

    """
    w = np.load(input_file)

    dim = dimenstion

    selected_dim = w[dim, :]

    objects = range(0, len(selected_dim), 10)
    y_pos = np.arange(len(selected_dim))

    plt.bar(y_pos, selected_dim, align='center', alpha=0.5)
    plt.ylabel('Weight')
    plt.xlabel('Categories')
    plt.title(f'Categorical Decomposition of {dim}. dimension')

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Categorical Decomposition')

    # Required parameters
    parser.add_argument("input_file", type=str,
                        help="Bhattacharya distance matrix (w_b.npy)")
    parser.add_argument("dimension", type=int,
                        help="The dimension of decomposition")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    input_file = args.input_file
    dimension = args.dimension
    output_file = args.output_file

    plot_categorical_decomposition(input_file, dimension, output_file)