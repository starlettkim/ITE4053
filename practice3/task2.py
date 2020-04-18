from typing import Tuple

import nn
import numpy as np


def generate_data(low: int, high: int,
                  shape: Tuple[int, int]) \
        -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randint(low, high, shape)
    y = (X.sum(axis=0, keepdims=True) > 0).astype(int)
    return X, y


if __name__ == '__main__':
    train_X, train_y = generate_data(-10, 11, (2, 1000))
    test_X, text_y = generate_data(-10, 11, (2, 100))

    model = nn.model.Sequential([
        nn.layers.Dense(2, 1, nn.activations.Sigmoid)
    ])
    loss = nn.loss.BCE()

    for _ in range(100):
        model.fit(train_X, train_y, loss, lr=1e-3)
    print(np.mean(np.round(model.forward(train_X)) == train_y))

'''
    iters = 100
    for i in range(iters):
        model.fit(train_X, train_y)
        print(model.evaluate())
'''