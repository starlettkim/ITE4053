from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, GaussianNoise, Conv2D, Add
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np


def main(train=False, test_img_path=None):
    # Load data
    (x_train, _), (x_test, _) = cifar10.load_data()
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
    x = Add()([x, input_tensor])
    model = Model(input_tensor, x)
    model.summary()

    # Train
    if train:
        model.compile(optimizer='Adam', loss='mse')
        model.fit(x_train, y_train, batch_size=32, epochs=100)
        model.save_weights('./checkpoints/model2')
    else:
        pass
        model.load_weights('./checkpoints/model2')

    # Test
    if test_img_path is not None:
        model.layers.pop(0)
        test_img = load_img(test_img_path)
        test_img = img_to_array(test_img)
        for i in range(0, test_img.shape[0], 32):
            for j in range(0, test_img.shape[1], 32):
                test_img[i:i + 32, j:j + 32] = model.predict(np.expand_dims(test_img[i:i + 32, j:j + 32], 0))
        test_img = array_to_img(test_img)
        test_img.save('Model2.png')


if __name__ == '__main__':
    main(True, test_img_path='noisy.png')
