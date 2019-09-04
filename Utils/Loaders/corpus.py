import os
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def read(input_file, max_words=-1):
    """
    Reads in corpus
    Parameters
    ----------
    input_file: str
        Input file
    max_words: int
        Maximum words to read

    Returns
    -------
    list:
        List of the words
    """
    with open(input_file, mode='r', encoding='utf8') as file:
        line = True
        counter = 0
        words = []

        while line:
            line = file.readline()
            if line.startswith("#!comment:"):
                continue
            line = line.strip()
            words.append(line)
            counter += 1
            if counter == max_words:
                break

        logging.info(f"{counter} words are read in from "
                     f"'{os.path.join(os.getcwd(), input_file)}' corpus.")

        return words
