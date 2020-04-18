import numpy as np
import typing


class Loss(object):
    def __init__(self):
        pass

    @staticmethod
    def forward(y: np.ndarray,
                y_hat: np.ndarray) \
            -> float:
        pass

    @staticmethod
    def backward(y: np.ndarray,
                 y_hat: np.ndarray) \
            -> np.ndarray:
        pass


class BCE(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y: np.ndarray,
                y_hat: np.ndarray) \
            -> float:
        y_hat[y_hat == 0] += 2**-53
        y_hat[y_hat == 1] -= 2**-53
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    @staticmethod
    def backward(y: np.ndarray,
                 y_hat: np.ndarray) \
            -> np.ndarray:
        y_hat[y_hat == 0] += 2 ** -53
        y_hat[y_hat == 1] -= 2 ** -53
        return -(y / y_hat - (1 - y) / y_hat)
