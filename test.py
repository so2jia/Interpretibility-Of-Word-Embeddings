from Eval.glove_funcs import score_2 as score
from Utils.Loaders.semcat import read as semcat_reader
from Utils.Loaders.embedding import read as embedding_reader
import numpy as np

embedding = embedding_reader("out/I.embedding.100d.txt", dense_file=True, lines_to_read=50000)
semcat = semcat_reader("data/semcat/Categories/")
I = np.load("out/e_s.npy")
w_b = np.load("out/w_b.npy")

# s = score.score(embedding.W, embedding, semcat, w_b, lamb=10, norm=False)
# print(s)