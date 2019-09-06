from Utils.Loaders import embedding as loader
from Utils.Loaders import semcat as sc
from Eval.glove_funcs.bhattacharya import bhattacharya_matrix
from Eval.glove_funcs.score import score

from sklearn.preprocessing import StandardScaler

import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def main():
    # Reading files
    embedding = loader.read("../data/glove/glove.6B.300d.txt", True, lines_to_read=50000)
    semcat = sc.read("../data/semcat/Categories")

    # Calculating Bhattacharya distance
    W_b, W_bs = bhattacharya_matrix(embedding, semcat, save=True, load=False)

    # Normalized matrix
    logging.info("Normalizing Bhattacharya matrix...")
    W_nb = W_b / np.linalg.norm(W_b, 1, axis=0)

    # Sign corrected matrix
    logging.info("Performing sign correction...")
    W_nsb = W_nb * W_bs

    # Standardize epsilon
    logging.info("Standardising embedding vectors...")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(embedding.W)
    epsilon_s = scaler.transform(embedding.W)

    # Calculating
    I = epsilon_s.dot(W_nsb)

    # Scoring
    logging.info("Calculating scores...")
    s = score(embedding, I, semcat, 1)
    logging.info(f"Score: {s}")


if __name__ == '__main__':
    main()
