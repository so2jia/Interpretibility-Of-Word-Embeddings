import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def debug(func):
    def _decorator(self, *args, **kwargs):
        func(self, *args, **kwargs)
        if self.debug:
            logging.debug(f"After {self.__class__.__name__} the weights are:")
            logging.debug("\n"+str(self.array))

    return _decorator
