from typing import Optional, Type

import numpy as np
from .activations import *


class Layer(object):
    def __init__(self,
                 input_dim: int, output_dim: int,
                 activation: Optional[Type[Activation]] = None,
                 dropout_rate: Optional[int] = 0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if activation is not None:
            self.activation = activation()
        self.dropout_rate = dropout_rate

    def forward(self, x: np.ndarray) \
            -> np.ndarray:
        pass

    def backward(self, grad: np.ndarray) \
            -> np.ndarray:
        pass

    def update(self, lr: float) \
            -> None:
        pass

    def init(self) \
            -> None:
        pass


class Dense(Layer):
    def __init__(self,
                 input_dim: int, output_dim: int,
                 activation: Optional[Type[Activation]] = None,
                 dropout_rate: Optional[int] = 0):
        super().__init__(input_dim, output_dim,
                         activation,
                         dropout_rate)
        self.dw = None
        self.db = None
        self.x = None
        self.weights = None
        self.bias = None
        self.init()

    def forward(self, x: np.ndarray) \
            -> np.ndarray:
        self.x = x
        y = np.dot(self.weights, x) + self.bias
        if hasattr(self, 'activation'):
            y = self.activation.forward(y)
        return y

    def backward(self, grad: np.ndarray) \
            -> np.ndarray:
        if hasattr(self, 'activation'):
            grad = self.activation.backward(grad)
        self.dw = np.dot(grad, self.x.T) / self.x.shape[-1]
        self.db = grad.sum(axis=-1, keepdims=True) / self.x.shape[-1]
        return np.dot(self.weights.T, grad)

    def update(self, lr: float) \
            -> None:
        self.weights -= lr * self.dw
        self.bias -= lr * self.db

    def init(self) \
            -> None:
        self.weights = np.random.randn(self.output_dim, self.input_dim)
        self.bias = np.random.randn(self.output_dim, 1)
