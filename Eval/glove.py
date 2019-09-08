from Utils.Loaders import embedding as loader
from Utils.Loaders import semcat as sc
from Eval.glove_funcs.bhattacharya import bhattacharya_matrix
from Eval.glove_funcs.score import score

from sklearn.preprocessing import StandardScaler

import numpy as np
from numpy import ndarray
from argparse import ArgumentParser
import os
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class Glove:
    def __init__(self, embedding_path, semcat_dir, weights_dir="out", save=True, load=False):
        # Reading files
        self.embedding = loader.read(embedding_path, True, lines_to_read=50000)
        self.semcat = sc.read(semcat_dir)

        self.weights_dir = weights_dir
        self.save = save
        self.load = load

        self.output = None

        self._eval()

    def _eval(self):
        # Calculating Bhattacharya distance
        W_b, W_bs = bhattacharya_matrix(self.embedding, self.semcat, output_dir=self.weights_dir, save=self.save, load=self.load)

        # Normalized matrix
        logging.info("Normalizing Bhattacharya matrix...")
        W_nb = W_b / np.linalg.norm(W_b, 1, axis=0)

        # Sign corrected matrix
        logging.info("Performing sign correction...")
        W_nsb = W_nb * W_bs

        # Standardize epsilon
        logging.info("Standardising embedding vectors...")
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(self.embedding.W)
        epsilon_s = scaler.transform(self.embedding.W)

        # Calculating
        I = epsilon_s.dot(W_nsb)

        if self.save:
            prefix = os.path.join(os.getcwd(), self.weights_dir)
            np.save(os.path.join(prefix, 'I.npy'), I)

        self.output = I

    def calculate_score(self, lamb=1):
        if self.output is None:
            logging.info("Eval not called!")
            return None
        # Scoring
        logging.info("Calculating score...")
        s = score(self.embedding, self.output, self.semcat, lamb)
        logging.info(f"Score with lambda={lamb} => {s}")
        return s

    def calculate_semantic_decomposition(self, word: str, top=20):
        """
        Calculating semantic decomposition of a word
        Parameters
        ----------
        word: str
            The word to decompose
        top: int
            The number of top categories to get

        Returns
        -------
        ndarray:
            An array where the first vector contains the category/dimension ID and the second the weight
        """
        logging.info(f"Semantic decomposition of word: {word}")
        logging.info("=========================================")

        # IDs of categories which containing the word
        word_categories = []
        for cat in self.semcat.vocab:
            if word in self.semcat.vocab[cat]:
                word_categories.append(self.semcat.c2i[cat])

        category_dims = self.output[self.embedding.w2i[word], :]

        # creating dimension - weight matrix
        cid = []
        for k, v in enumerate(category_dims):
            cid.append([k, v])

        cid = np.array(cid)

        # sorting by weights
        cid_sorted = cid[cid[:, 1].argsort()]

        # printing out the top 20 category
        for vec in cid_sorted[-top:, :]:
            if vec[0] in word_categories:
                logging.info(f"> {self.semcat.i2c[vec[0]]}: {vec[1]}")
            else:
                logging.info(f"{self.semcat.i2c[vec[0]]}: {vec[1]}")

        return cid_sorted[-top:, :]
