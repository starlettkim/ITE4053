from typing import Tuple, Iterable
import numpy as np


def generate_data(low: int, high: int,
                  shape: Iterable[int]) \
        -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.randint(low, high, shape)
    y = (x[0] * x[0] > x[1]).reshape((1, shape[1])).astype(int)
    return x, y
