import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class SemCat:
    """
    Wraps vocab and converter dictionaries
    """
    def __init__(self, vocab, c2i, i2c):
        self._vocab = vocab
        self._c2i = c2i
        self._i2c = i2c

    @property
    def vocab(self):
        """
        Returns a dictionary where keys are the categories and the values are the lists of the words related to them
        Returns
        -------
        dict:
            key -> [List]
        """
        return self._vocab

    @property
    def c2i(self):
        """
        Category name to index dictionary
        Returns
        -------
        dict:
            key -> value
        """
        return self._c2i

    @property
    def i2c(self):
        """
        Index to category name dictionary
        Returns
        -------
        dict:
            key -> value
        """
        return self._i2c


def read(input_dir: str):
    """
    Reads in SEMCAT categories and words

    Parameters
    ----------
    input_dir: str
        The path to the directory where the categories are found.

    Returns
    -------
    SemCat:
        Wrapper
    """
    vocab = {}
    vocab_size = 0

    w2i, i2w = {}, {}

    id = 0

    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            category_name = file.rstrip('.txt').split('-')[0]
            with open(os.path.join(input_dir, file), mode='r', encoding='utf8') as f:
                words = f.read().splitlines()
                vocab_size += words.__len__()
                w2i[category_name] = id
                vocab[category_name] = words
                id += 1

    i2w = {v: k for k, v in w2i.items()}

    logging.info(
        f"{vocab.__len__()} categories are read from SEMCAT files, which contain overall {vocab_size} words.")
    return SemCat(vocab, w2i, i2w)
