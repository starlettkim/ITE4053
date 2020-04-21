from typing import Tuple, Iterable
import numpy as np


def generate_data(low: int, high: int,
                  shape) \
        -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.rand(*shape) * (high - low) + low
    y = (x[0] * x[0] > x[1]).reshape((1, shape[1])).astype(int)
    return x, y
