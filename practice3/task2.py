import test
import nn

model = nn.models.Sequential([
    nn.layers.Dense(2, 1, nn.activations.Sigmoid),
    nn.layers.Dense(1, 1, nn.activations.Sigmoid)
])

if __name__ == '__main__':
    test.test_models([model],
                     train_shape=(2, 1000),
                     test_shape=(2, 100),
                     lr=1, epochs=1000, num_run=100)
