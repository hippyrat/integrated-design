import numpy as np

class ops:
    @staticmethod
    def MatMul(a, b):
        return np.dot(a.value, b.value)

    @staticmethod
    def Add(a, b):
        return a + b

    class Step:
        def __init__(self, output):
            self.output = output

        def forward(self):
            self.value = np.where(self.output > 0, 1, -1)

class loss:
    @staticmethod
    def PerceptionLoss(predicted, actual):
        return np.mean((predicted - actual) ** 2)