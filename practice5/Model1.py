from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *


def main():
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    y_train, y_test = x_train, x_test

    # Construct model
    input_tensor = Input(shape=(32, 32, 3))
    x = GaussianNoise(0.1)(input_tensor)
    for _ in range(4):
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(3, 3, padding='same')(x)
    model = keras.Model(input_tensor, x)
    model.summary()

    # Train
    model.compile(optimizer='Adam', loss='mse')
    model.fit(x_train, y_train, batch_size=32, epochs=100)


if __name__ == '__main__':
    main()
