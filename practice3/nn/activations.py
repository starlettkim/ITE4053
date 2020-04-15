import numpy as np


class Activation(object):
    def __init__(self):
        pass

    def forward(selfself, x: np.ndarray) \
            -> np.ndarray:
        pass

    def backward(self, grad: np.ndarray) \
            -> np.ndarray:
        pass


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.y = np.ndarray

    def forward(self, x: np.ndarray) \
            -> np.ndarray:
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, grad: np.ndarray) \
            -> np.ndarray:
        return self.y * (1 - self.y) * grad
