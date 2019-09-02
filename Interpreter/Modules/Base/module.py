from Interpreter.Modules.Abstract.abstractmodule import AbstractModule
import numpy as np

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class Module(AbstractModule):
    def __init__(self, debug=False):
        self.debug = debug
        self.array = np.zeros(1)