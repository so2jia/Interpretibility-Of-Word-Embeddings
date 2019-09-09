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
    def __init__(self, embedding_params: dict, semcat_dir, eval_params, calculation_type, calculation_args):
        """

        Parameters
        ----------
        embedding_params: dict
            {
            "input_file": embedding_path,
            "dense_file": dense_file,
            "lines_to_read": lines_to_read,
            "mcrae_dir": mcrae_dir,
            "mcrae_words_only": mcrae_words_only
            }
        semcat_dir: str
            Path to SemCat Categories directory
        eval_params: dict
            {
                "weights_dir": "out/",
                "save_weights": False,
                "load_weights": True
            }
        calculation_type: str
            ['score'|'decomp']
        calculation_args: list
            *args
        """
        if embedding_params["dense_file"] is None:
            embedding_params["dense_file"] = False
        if embedding_params["lines_to_read"] is None:
            embedding_params["lines_to_read"] = -1

        if embedding_params["mcrae_words_only"] is None:
            embedding_params["mcrae_words_only"] = False

        if eval_params["weights_dir"] is None:
            eval_params["weights_dir"] = "out"

        if eval_params["save_weights"] is None:
            eval_params["save_weights"] = False
        if eval_params["load_weights"] is None:
            eval_params["load_weights"] = False

        # Reading files
        self.embedding = loader.read(**embedding_params)
        self.semcat = sc.read(semcat_dir)

        self.eval_params = eval_params
        self.eval_params["embedding"] = self.embedding
        self.eval_params["semcat"] = self.semcat

        self.calc_types = {
            "score": self.calculate_score,
            "decomp": self.calculate_semantic_decomposition
        }

        self.output = None
        if embedding_params["dense_file"]:
            self._eval()
        else:
            logging.info("Sparse model is not implemented yet!")

        if calculation_type in self.calc_types.keys():
            self.calc_types[calculation_type](*calculation_args)
        else:
            logging.info("Wrong calculation type provided or not provided at all!")

    def _eval(self):
        # Calculating Bhattacharya distance
        W_b, W_bs = bhattacharya_matrix(**self.eval_params)

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

        if self.eval_params["save_weights"]:
            prefix = os.path.join(os.getcwd(), self.eval_params["weights_dir"])
            np.save(os.path.join(prefix, 'I.npy'), I)
            np.save(os.path.join(prefix, 'w_nb.npy'), W_nb)
            np.save(os.path.join(prefix, 'w_nsb.npy'), W_nsb)
            np.save(os.path.join(prefix, 'e_s.npy'), epsilon_s)

        self.output = I

    def calculate_score(self, lamb=1):
        if type(lamb) != int:
            try:
                lamb = int(lamb)
            except TypeError:
                logging.info("Lambda value is not an integer or can be converted to it!")
                return None

        if lamb < 1:
            logging.info("Lambda must be greater or equal to 1")

        if self.output is None:
            logging.info("Eval not called!")
            return None
        # Scoring
        logging.info("Calculating score...")
        s = score(self.embedding, self.output, self.semcat, lamb)
        logging.info(f"Score with lambda={lamb} => {s}")
        return s

    def calculate_semantic_decomposition(self, word: str, top=20, save=False):
        """
        Calculating semantic decomposition of a word
        Parameters
        ----------
        word: str
            The word to decompose
        top: int
            The number of top categories to get
        save: bool
            Save to file
        Returns
        -------
        ndarray:
            An array where the first vector contains the category/dimension ID and the second the weight
        """
        if type(word) != str:
            word = str(word)

        if type(top) != int:
            try:
                top = int(top)
            except TypeError:
                logging.info("Calculation param: TOP can not be converted to int")
                return None

        if type(save) != bool:
            try:
                save = bool(save)
            except TypeError:
                logging.info("Calculation param: SAVE can not be converted to bool")
                return None

        if self.output is None:
            logging.info("Eval not called!")
            return None

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
        top_values = cid_sorted[-top:, :]
        for vec in top_values:
            if vec[0] in word_categories:
                logging.info(f"> {self.semcat.i2c[vec[0]]}: {vec[1]}")
            else:
                logging.info(f"{self.semcat.i2c[vec[0]]}: {vec[1]}")

        if save:
            with open(f"out/decom-{word}.txt", mode="w", encoding="utf8") as f:
                for vec in top_values:
                    f.write(f"{self.semcat.i2c[vec[0]]} {vec[0]} {vec[1]}\n")

        return top_values

    def get_calculation_types(self):
        return self.calc_types.keys()
