import numpy as np
from Interpreter.Modules.Base.module import Module
from Interpreter.Modules.Base.debug import debug


class Normalize(Module):
    @debug
    def forward(self, array: np.ndarray) -> np.ndarray:
        self.array = array / np.linalg.norm(array, ord=1, axis=1)
        return self.array
