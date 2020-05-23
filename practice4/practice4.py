from typing import Tuple
import numpy as np
import tensorflow as tf
import time
from tqdm import trange


def generate_data(low: int, high: int,
                  shape) \
        -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(low, high, shape)
    y = np.expand_dims(x[:,0] ** 2 > x[:,1], 1).astype(int)
    return x, y


if __name__ == '__main__':

    num_runs = 20

    train_elapsed = 0
    train_loss = 0
    train_acc = 0

    test_elapsed = 0
    test_loss = 0
    test_acc = 0

    for _ in trange(num_runs):
        train_x, train_y = generate_data(-2, 2, (1000, 2))
        test_x, test_y = generate_data(-2, 2, (100, 2))

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(3, input_shape=(2,), activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(.5),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=[tf.keras.metrics.binary_accuracy])

        train_elapsed -= time.time() / num_runs
        result = model.fit(train_x, train_y,
                  batch_size=1000,
                  epochs=1000,
                  verbose=0)
        train_elapsed += time.time() / num_runs
        train_loss += result.history['loss'][-1] / num_runs
        train_acc += result.history['binary_accuracy'][-1] / num_runs

        test_elapsed -= time.time() / num_runs
        result = model.evaluate(test_x, test_y, verbose=0)
        test_elapsed += time.time() / num_runs
        test_loss += result[0] / num_runs
        test_acc += result[1] / num_runs

    print('# Number of Runs: %d\n' % num_runs)
    print('## Train: ')
    print('   loss: %f' % (train_loss))
    print('   accuracy: %f%%' % (train_acc * 100))
    print('   elapsed: %ss\n' % (train_elapsed))

    print('## Test: ')
    print('   loss: %f' % (test_loss))
    print('   accuracy: %f%%' % (test_acc * 100))
    print('   elapsed: %ss\n' % (test_elapsed))