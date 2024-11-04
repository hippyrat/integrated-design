import numpy as np
class Variable:
    def __init__(self, dim, init=False, trainable=False):
        self.dim = dim
        self.trainable = trainable
        self.value = np.random.rand(*dim) if init else None
        self.grad = np.zeros(dim) if trainable else None

    def set_value(self, value):
        self.value = value