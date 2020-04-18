from typing import Callable, List, Optional

import numpy as np
import nn


class Model(object):
    def __init__(self):
        pass

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            loss: Callable[[np.ndarray, np.ndarray], float]
            ) \
            -> None:
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
            loss: nn.Loss,
            lr: Optional[float] = 1e-6) \
            -> None:
        y_hat = self.forward(x)
        grad = loss.backward(y, y_hat)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            layer.update(lr)

    def evaluate(self, x: np.ndarray, y: np.ndarray) \
            -> float:
        y_hat = self.forward(x)
        return 1
