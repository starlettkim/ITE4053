import numpy as np


class Activation(object):
    def __init__(self):
        pass


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(-X))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return