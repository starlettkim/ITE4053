import numpy as np
import nn


class Metric(object):
    name = ''

    @classmethod
    def compute(cls,
                y: np.ndarray,
                y_hat: np.ndarray) \
            -> float:
        pass


class BinaryAccuracy(Metric):
    name = 'BinaryAccuracy'

    @classmethod
    def compute(cls,
                y: np.ndarray,
                y_hat: np.ndarray) \
            -> float:
        return (y == np.round(y_hat)).mean()


class BinaryCrossentropy(Metric):
    name = 'BinaryClassEntropy'

    @classmethod
    def compute(cls,
                y: np.ndarray,
                y_hat: np.ndarray) \
            -> float:
        return nn.losses.BinaryCrossentropy.compute(y, y_hat).mean()
