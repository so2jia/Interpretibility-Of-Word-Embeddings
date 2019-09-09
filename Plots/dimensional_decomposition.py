import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from Utils.Loaders.semcat import read


def plot_categorical_decomposition(semcat_dir, input_file, category, output_file=None):
    """
    Plot categorical decomposition
    Parameters
    ----------
    semcat_dir: str
        Path to SemCat categories directory
    input_file: str
        Path to Category-Weight matrix (w_b.npy)
    dimenstion: int
        The dimension of decomposition
    output_file
        Output PNG file (Optional)
    Returns
    -------

    """
    semcat = read(semcat_dir)
    w = np.load(input_file)

    selected_dim = w[:, semcat.c2i[category]]

    y_pos = np.arange(len(selected_dim))

    plt.bar(y_pos, selected_dim, align='center', alpha=0.5)
    plt.ylabel('Weight')
    plt.xlabel('Dimensions')
    plt.title(f'Dimensional Decomposition of {category} category')

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Dimensional Decomposition')

    # Required parameters
    parser.add_argument("semcat_dir", type=str,
                        help="Path to SemCat categories directory")
    parser.add_argument("input_file", type=str,
                        help="Bhattacharya distance matrix (w_b.npy)")
    parser.add_argument("category", type=str,
                        help="The category of decomposition")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    semcat_dir = args.semcat_dir
    input_file = args.input_file
    category = args.category
    output_file = args.output_file

    plot_categorical_decomposition(semcat_dir, input_file, category, output_file)