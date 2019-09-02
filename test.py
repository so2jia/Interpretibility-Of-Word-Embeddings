from Utils.Loaders import embedding as loader
from Utils.Loaders import semcat as sc
from Interpreter.pipeline import Pipeline
import numpy as np

from Eval import glove

from Interpreter.Modules.Normalize import Normalize

# embedding = loader.read("data/glove/glove.6B.300d.txt", True, lines_to_read=50000)
# semcat = sc.read("data/semcat/Categories")
#
# w2i = embedding.w2i
# i2w = embedding.i2w
#
# w: np.ndarray
# W = embedding.W

inp = np.array([[1, 2, 5], [-2, 3, 7], [-1, 0, 2]])

m = Normalize(debug=True)

interpreter = Pipeline()
interpreter.add(m)

res = interpreter.run(inp)
