from typing import Callable, List

import numpy as np
import nn


class Model(object):
    def __init__(self):
        self.loss = Callable[[np.ndarray, np.ndarray], float]
        pass

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            loss: Callable[[np.ndarray, np.ndarray], float]
            ) \
            -> None:
        self.loss = loss
        pass

    # Evaluate model on dataset.
    def evaluate(self, x: np.ndarray, y: np.ndarray) \
            -> float:
        pass


class Sequential(Model):
    def __init__(self,
                 layers: List[nn.layers.Layer]):
        super().__init__()
        self.layers = layers

    def forward(self, x: np.ndarray) \
            -> np.ndarray:
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            loss: Callable[[np.ndarray, np.ndarray], float]) \
            -> None:
        pass
        '''
        y_hat = self.forward(x)
        l = self.loss(y_hat, y)
        for layer in reversed(self.layers):
            l = layer.backward(l)
            layer.update()
            '''

    def evaluate(self, x: np.ndarray, y: np.ndarray) \
            -> float:
        y_hat = self.forward(x)
        return self.loss(y_hat, y)
