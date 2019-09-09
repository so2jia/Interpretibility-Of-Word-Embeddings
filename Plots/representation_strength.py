import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from Utils.Loaders.semcat import read

import math

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def plot_representation_strength(input_file, semcat_dir, output_file=None):
    w = np.load(input_file)
    semcat = read(semcat_dir)

    rep = np.sum(w, axis=0)
    rep = np.append([rep], [np.arange(len(rep))], axis=0)
    arsort = rep[0, :].argsort()
    rep_sorted = rep[:, arsort]

    y_pos = np.arange(len(rep[0]))

    plt.bar(y_pos, rep_sorted[0, ::-1], align='center', alpha=0.5)
    plt.ylabel('Representation Strength')
    plt.xlabel('Category')

    for i in range(rep_sorted.shape[1]-1, -1, -1):
        logging.info(f"#{math.fabs(i-rep_sorted.shape[1])} {semcat.i2c[rep_sorted[1, i]]} => {rep_sorted[0, i]}")

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Representation Strength')

    # Required parameters
    parser.add_argument("input_file", type=str,
                        help="Bhattacharya distance matrix (w_b.npy)")
    parser.add_argument("semcat_dir", type=str,
                        help="Path to SemCat categories directory")
    parser.add_argument("--output_file", type=str,
                        help="Output PNG file (Optional)")

    args = parser.parse_args()

    input_file = args.input_file
    semcat_dir = args.semcat_dir
    output_file = args.output_file

    plot_representation_strength(input_file, semcat_dir, output_file)