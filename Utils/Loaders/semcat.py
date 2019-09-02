import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def read(input_dir: str):
    vocab = {}
    vocab_size = 0
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            category_name = file.rstrip('.txt').split('-')[0]
            with open(os.path.join(input_dir, file), mode='r') as f:
                words = f.read().splitlines()
                vocab_size += words.__len__()
                vocab[category_name] = words
    logging.info(f"{vocab.__len__()} categories are read from SEMCAT files.")
    return vocab
