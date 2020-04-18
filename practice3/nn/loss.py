import numpy as np
import typing


class Loss(object):
    def __init__(self):
        pass


class BCE(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(self,
                y: np.ndarray,
                y_hat: np.ndarray) \
            -> float:
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    @staticmethod
    def backward(y: np.ndarray,
                 y_hat: np.ndarray) \
            -> np.ndarray:
        return -(y / y_hat - (1 - y) / y_hat)
