from typing import Tuple
import numpy as np


def generate_data(low: int, high: int,
                  shape: Tuple[int, int]) \
        -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randint(low, high, shape)
    y = (X[0] * X[0] > X[1]).reshape((1, shape[1])).astype(int)
    return X, y


train_x, train_y = generate_data(-2, 2, (2, 1000))
test_x, test_y = generate_data(-2, 2, (2, 100))
