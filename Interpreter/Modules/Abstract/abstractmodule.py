import abc
import numpy as np

from Interpreter.Modules.Base.debug import debug


class AbstractModule(abc.ABC):
    @debug
    def forward(self, array: np.ndarray) -> np.ndarray:
        pass