import numpy as np
from Utils.Loaders.embedding import read as embreader, Embedding
import os

def _save_embedding(I: np.ndarray, embedding: Embedding, prefix, name):
    with open(os.path.join(prefix, f'{name}_I.embedding.100d.txt'), mode='w', encoding='utf8') as f:
        for i in range(I.shape[0]):
            f.write(f"{embedding.i2w[i]}")
            for j in range(I.shape[1]):
                f.write(f" {I[i, j]}")
            f.write("\n")

emb = embreader(f"../../data/glove/glove.6B.300d.txt", True, 50000)
e_s = np.load(f"../../out/kde/dense_model/raw_e_s.npy")
w_nsb = np.load(f"../../out/kde/dense_model/raw_w_nsb.npy")

I = e_s.dot(w_nsb)

prefix = os.path.join(os.getcwd(), "../../out/kde/dense_model/")

_save_embedding(I, emb, prefix, f"raw")

# Sparse
# regul = "5"
# emb = embreader(f"../../data/glove/sparse/glove300d_l_0.{regul}_DL_top50000.emb.gz", False, 50000)
# e_s = np.load(f"../../out/kde/sparse_model/l_0.{regul}_e_s.npy")
# w_nsb = np.load(f"../../out/kde/sparse_model/l_0.{regul}_w_nsb.npy")
#
# I = e_s.dot(w_nsb)
#
# prefix = os.path.join(os.getcwd(), "../../out/kde/sparse_model/")
#
# _save_embedding(I, emb, prefix, f"l_0.{regul}")
