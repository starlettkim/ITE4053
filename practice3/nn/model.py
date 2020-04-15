from typing import Callable, List

import numpy as np
import nn


class Model(object):
    def __init__(self, loss: Callable[[np.ndarray, np.ndarray], float]):
        self.loss = loss
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    # Evaluate model on dataset.
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        pass


class Sequential(Model):
    def __init__(self, layers: List[nn.Layer], loss: Callable[[np.ndarray, np.ndarray], float]):
        super().__init__(loss)
        self.layers = layers

    def __forward__(self, X: np.ndarray):
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_hat = self.__forward__(X)
        l = self.loss(y_hat, y)
        for layer in reversed(self.layers):
            l = layer.backward(l)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        y_hat = self.__forward__(X)
        return self.loss(y_hat, y)
