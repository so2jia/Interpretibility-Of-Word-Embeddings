from Interpreter.Modules.Base.module import Module


class Pipeline:
    def __init__(self):
        self.pipeline = []

    def add(self, module: Module):
        self.pipeline.append(module)

    def run(self, input_array):
        array = input_array
        module: Module
        for module in self.pipeline:
            if isinstance(module, Module):
                array = module.forward(array)
        return array
