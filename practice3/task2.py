import nn
import numpy as np


def generate_data(low: int, high: int, shape: tuple):
    X = np.random.randint(low, high, shape)
    y = (np.sum(X, 0) > 0).astype(int)
    return X, y


if __name__ == '__main__':
    model = nn.Sequential(
        [nn.layers.Dense(2, 1)]
    )

    train_X, train_y = generate_data(-2, 3, (2, 1000))
    test_X, text_y = generate_data(-2, 3, (2, 100))

    iters = 100
    for i in range(iters):
        model.fit(train_X, train_y)
        print(model.evaluate())
