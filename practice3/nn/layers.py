from typing import Optional

import numpy as np
from .activations import *


class Layer(object):
    def __init__(self,
                 input_dim: int, output_dim: int,
                 activation: Optional[Activation] = None,
                 dropout_rate: Optional[int] = 0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.dW, self.db = 0, 0
        self.y = 0

    def forward(self, x: np.ndarray) \
            -> np.ndarray:
        pass

    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass

    def update(self):
        pass


class Dense(Layer):
    def __init__(self,
                 input_dim: int, output_dim: int,
                 activation: Optional[type(Activation)] = None,
                 dropout_rate: Optional[int] = 0):
        super().__init__(input_dim, output_dim,
                         activation,
                         dropout_rate)

        self.weights = np.random.randn(output_dim, input_dim) * .1
        self.bias = np.random.randn(output_dim, 1) * .1

    def forward(self, x: np.ndarray) \
            -> np.ndarray:
        self.y = np.dot(self.weights, x) + self.bias
        return self.y

    def backward(self, grad: np.ndarray) \
            -> np.ndarray:
        pass

    def update(self) \
            -> None:
        pass
