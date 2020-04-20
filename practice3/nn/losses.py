import numpy as np
import nn.metrics


class Loss(nn.Metric):
    def __init__(self):
        pass

    @classmethod
    def compute(cls,
                y: np.ndarray,
                y_hat: np.ndarray) \
            -> np.ndarray:
        pass

    @classmethod
    def backward(cls,
                 y: np.ndarray,
                 y_hat: np.ndarray) \
            -> np.ndarray:
        pass


class BinaryCrossentropy(Loss):
    def __init__(self):
        super().__init__()

    @classmethod
    def compute(cls,
                y: np.ndarray,
                y_hat: np.ndarray) \
            -> np.ndarray:
        y_hat[y_hat == 0] += 2**-53
        y_hat[y_hat == 1] -= 2**-53
        return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    @classmethod
    def backward(cls,
                 y: np.ndarray,
                 y_hat: np.ndarray) \
            -> np.ndarray:
        y_hat[y_hat == 0] += 2 ** -53
        y_hat[y_hat == 1] -= 2 ** -53
        return -y / y_hat + (1 - y) / (1 - y_hat)
