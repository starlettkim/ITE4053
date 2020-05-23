from typing import Tuple
import numpy as np
import tensorflow as tf


def generate_data(low: int, high: int,
                  shape) \
        -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.rand(*shape) * (high - low) + low
    y = np.expand_dims(x[:,0] * x[:,0] > x[:,1], 1).astype(int)
    return x, y


if __name__ == '__main__':
    train_x, train_y = generate_data(-2, 2, (1000, 2))
    test_x, test_y = generate_data(-2, 2, (100, 2))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, input_shape=(2,), activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_accuracy])

    model.fit(train_x, train_y,
              batch_size=1000,
              epochs=1000,
              verbose=0)

    print('Evaluation on Train: ')
    result = model.evaluate(train_x, train_y, verbose=0)
    print(dict(zip(model.metrics_names, result)))

    print()
    print('Evaluation on Test: ')
    result = model.evaluate(test_x, test_y, verbose=0)
    print(dict(zip(model.metrics_names, result)))
