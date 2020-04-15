import numpy as np


class Model(object):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def evaluate(self):
        pass


class Sequential(Model):
    def __init__(self, layers: []):
        super().__init__()
        self.layers = layers

    def fit(self, X: np.ndarray, y: np.ndarray):
        for layer in self.layers:
            X = layer.forward(X)

    def evaluate(self):
        pass