import numpy as np

np.random.seed(1)
N = 20
X = np.concatenate(
    (
        np.random.normal(0, 1, int(0.3 * N)),
        np.random.normal(5, 1, int(0.7 * N))
    )
)
print("")