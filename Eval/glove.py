from Utils.Loaders import embedding as loader
from Utils.Loaders import semcat as sc
from Utils.Loaders import corpus as wiki

import numpy as np

embedding = loader.read("../data/glove/glove.6B.300d.txt", True, lines_to_read=10000)
semcat = sc.read("../data/semcat/Categories")
corpus = wiki.read("../data/corpus/enwiki/wiki-100k.txt", 10000)

epsilon = []
w2e = {}
index = 0
unk_identifier = '<unk>'

for word in corpus:
    try:
        epsilon.append(embedding.W[embedding.w2i[word]])
        w2e[word] = index
    except KeyError:
        epsilon.append(embedding.W[embedding.w2i[unk_identifier]])
        w2e[unk_identifier] = index
    index += 1

epsilon = np.array(epsilon)

# W_b Matrix
W_b = np.zeros([embedding.W.shape[1], semcat.vocab.__len__()])

for i in range(W_b.shape[0]):
    for j in range(W_b.shape[1]):
        _p = []
        _q = []
        for word in semcat.vocab[semcat.i2c[j]]:
            try:
                _p.append(epsilon[w2e[word]][i])
            except KeyError:
                _p.append(epsilon[w2e[unk_identifier]][i])


def bhatta_distance(p, q):
    # Variance of p and q
    var1 = np.std(p) ** 2
    var2 = np.std(q) ** 2

    # Mean of p and q
    mean1 = np.mean(p)
    mean2 = np.mean(q)

    # Formula
    bc = np.log1p((var1 / var2 + var2 / var1 + 2) / 4) / 4 + ((mean1 - mean2) ** 2 / (var1 + var2)) / 4
    return bc

