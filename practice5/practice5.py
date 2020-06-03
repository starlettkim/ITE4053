import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam


def model1():
    return tf.keras.models.Sequential([
        tf.keras.layers.GaussianNoise(0.1, input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(3, 3, padding='same', activation='relu')
    ])


def main():
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    y_train, y_test = x_train, x_test

    m1 = model1()
    m1.compile(optimizer='Adam', loss='mse')
    m1.fit(x_train, y_train, batch_size=32, epochs=100)


if __name__ == '__main__':
    main()
