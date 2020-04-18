import nn
from data import train_x, train_y, test_x, test_y


if __name__ == '__main__':
    model = nn.model.Sequential([
        nn.layers.Dense(2, 1, nn.activations.Sigmoid),
        nn.layers.Dense(1, 1, nn.activations.Sigmoid)
    ])
    loss = nn.loss.BinaryCrossentropy

    model.fit(train_x, train_y, loss, lr=1e-2, epochs=1000)

    train_loss, train_acc = model.eval(train_x, train_y,
                                       [nn.metrics.BinaryCrossentropy, nn.metrics.BinaryAccuracy])
    print('> Train Set')
    print('Loss: %f' % train_loss)
    print('Accuracy: %f\n' % train_acc)

    test_loss, test_acc = model.eval(test_x, test_y,
                                     [nn.metrics.BinaryCrossentropy, nn.metrics.BinaryAccuracy])
    print('> Test Set')
    print('Loss: %f' % test_loss)
    print('Accuracy: %f\n' % test_acc)
