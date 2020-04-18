from typing import List, Optional, Type, Dict

import numpy as np
import nn


class Model(object):
    def __init__(self):
        pass

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            loss: Type[nn.Loss],
            lr: Optional[float],
            epochs: int) \
            -> None:
        pass


class Sequential(Model):
    def __init__(self,
                 layers: List[nn.Layer]):
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
            loss: Type[nn.Loss],
            lr: Optional[float] = 1e-6,
            epochs: int = 1) \
            -> None:
        for _ in range(epochs):
            y_hat = self.forward(x)
            grad = loss.backward(y, y_hat)
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
                layer.update(lr)

    def eval(self,
             x: np.ndarray, y: np.ndarray,
             metrics: List[Type[nn.Metric]]) \
            -> List[float]:
        y_hat = self.forward(x)
        result = []
        for metric in metrics:
            result.append(metric.compute(y, y_hat))
        return result
