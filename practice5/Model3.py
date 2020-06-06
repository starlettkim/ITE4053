from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers \
    import Input, Conv2D, Add, BatchNormalization, ReLU, Lambda
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import numpy as np


def main(train, eval, test_img_path=None, test_subtle=False):
    # Construct model
    input_tensor = Input(shape=(32, 32, 3))
    x = Lambda(lambda input_tensor: input_tensor)(input_tensor)
    for _ in range(4):
        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization(momentum=.0, epsilon=1e-4, axis=3)(x)
        x = ReLU()(x)
    x = Conv2D(3, 3, padding='same')(x)
    x = Add()([x, input_tensor])
    model = Model(input_tensor, x)

    lr_schedule = PiecewiseConstantDecay(
        [30 * 1563, 60 * 1563],
        [1e-4, 1e-5, 5e-6]
    )
    model.compile(optimizer=Adam(lr_schedule), loss='mse')

    # Train
    if train:
        # Load data
        (x_train, _), (_, _) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.
        y_train = x.train.copy()
        x_train += np.random.normal(0, .1, x_train.shape)
        model.fit(x_train, y_train, batch_size=32, epochs=100)
        model.save_weights('./checkpoints/model3')
    else:
        model.load_weights('./checkpoints/model3')

    if eval:
        # Load data
        (_, _), (x_test, _) = cifar10.load_data()
        x_test = x_test.astype('float32') / 255.
        y_test = x_test.copy()
        x_test += np.random.normal(0, .1, x_test.shape)
        print('Evaluating: ')
        model.evaluate(x_test, y_test)

    # Test
    if test_img_path is not None:
        test_img = load_img(test_img_path)
        test_img = img_to_array(test_img).astype(np.float32) / 255.
        new_img = np.zeros_like(test_img)

        if test_subtle:
            i_end = test_img.shape[0] - 16
            j_end = test_img.shape[1] - 16
            for i in range(0, i_end, 16):
                for j in range(0, j_end, 16):
                    predicted = model.predict(np.expand_dims(test_img[i:i+32, j:j+32], 0))
                    new_img[i+8*(i!=0) : i+32-8*(i!=i_end-16), j+8*(j!=0) : j+32-8*(j!=j_end-16)] \
                        = predicted[:, 8*(i!=0) : 32-8*(i!=i_end-16), 8*(j!=0) : 32-8*(j!=j_end-16)]
        else:
            for i in range(0, test_img.shape[0], 32):
                for j in range(0, test_img.shape[1], 32):
                    new_img[i:i+32, j:j+32] = model.predict(np.expand_dims(test_img[i:i+32, j:j+32], 0))

        new_img = array_to_img(new_img)
        new_img.save('data/Model3.png')
        new_img.show()


if __name__ == '__main__':
    main(False, True, 'data/noisy.png', True)
