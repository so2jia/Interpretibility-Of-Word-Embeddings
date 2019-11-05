import numpy as np

from Utils.Loaders.semcat import read as semcat_reader
from Utils.Loaders.embedding import read as embedding_reader
import os
from tqdm import trange
import tqdm
from Eval.InterpretabilityFunctions import is_v2_concept as interpretability

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def interpretability_scores(embedding_path, embedding_space, semcat_dir, distance_space, dense, lines_to_read,
                            lamb, norm, man_space, name, output):

    logging.info(f"Calculating score for {name}")

    path = output.split('/')
    os_path = os.getcwd()
    for dir in path[:-1]:
        os_path = os.path.join(os_path, dir)
        if not os.path.exists(os_path):
            os.mkdir(os_path)

    embedding = embedding_reader(embedding_path, dense, lines_to_read)
    semcat = semcat_reader(semcat_dir)

    w = np.load(distance_space)

    if man_space:
        ref_matrix = np.load(embedding_space)
    else:
        ref_matrix = embedding.W

    IS = []

    logging.info("Calculating interpretability...")
    for i in tqdm.tqdm([1, 5, 10]):
        IS.append(interpretability.score(ref_matrix, embedding, semcat, w, lamb=i, norm=norm, avg=True))

    # creating stats file complete path
    pp = os.path.join(os.getcwd(), output)
    # writing to file
    with open(pp, mode="w", encoding="utf8") as f:
        f.write(f"# Concept based scoring\n{name}\n# IS (Lamb [1, {lamb}]):\n")
        for s in IS:
            f.write(f"{s}\n")

    logging.info(f"IS^C for Lambda 10 is {IS[-1]}%")
