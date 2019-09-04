import numpy as np
from Interpreter.Modules.Base.module import Module


class Bhattacharya(Module):
    """
    Calculates the Bhattacharyya distance
    """

    @debug
    def forward(self, array: np.ndarray) -> np.ndarray:
        for i in range(array.shape[0]):
            self.bhatta_distance()
        self.array = array
        return self.array

    @classmethod
    def bhatta_distance(cls, p, q):
        # Variance of p and q
        var1 = np.std(p)**2
        var2 = np.std(q)**2

        # Mean of p and q
        mean1 = np.mean(p)
        mean2 = np.mean(q)

        bc = np.log1p((var1/var2 + var2/var1 + 2)/4)/4 + ((mean1 - mean2)**2 / (var1 + var2))/4
        return bc
