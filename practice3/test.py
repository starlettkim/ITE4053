import time
from typing import List, Dict, Iterable
from tqdm import trange

import nn
from data import generate_data
import task1
import task2
import task3


def test_models(models: List[nn.Model],
                train_shape: Iterable[int],
                test_shape: Iterable[int],
                lr: float,
                epochs: int,
                num_run: int) \
        -> None:
    results = [{}] * len(models)
    for i in range(len(models)):
        results[i] = {
            'elapsed_train': 0.,
            'elapsed_test': 0.,
            'loss_train': 0.,
            'loss_test': 0.,
            'acc_train': 0.,
            'acc_test': 0.
        }

    print()
    print('Running %d tests on %d model%s, with:' % (num_run, len(models), ('s' if len(models) > 1 else '')))
    print('# of train data: %s' % train_shape[-1])
    print('# of test data:  %s' % test_shape[-1])
    print('Learning rate:   %f' % lr)
    print('# of epochs:     %d\n' % epochs)

    t = trange(num_run)
    for _ in t:
        train_x, train_y = generate_data(-2, 2, train_shape)
        test_x, test_y = generate_data(-2, 2, test_shape)

        for idx, model in enumerate(models):
            model.init()

            start_time = time.time()
            model.fit(train_x, train_y, nn.losses.BinaryCrossentropy, lr, epochs)
            end_time = time.time()
            results[idx]['elapsed_train'] += float(end_time - start_time)

            loss_train, acc_train = model.eval(train_x, train_y,
                                               [nn.metrics.BinaryCrossentropy, nn.metrics.BinaryAccuracy])
            results[idx]['loss_train'] += loss_train
            results[idx]['acc_train'] += acc_train

            start_time = time.time()
            loss_test, acc_test = model.eval(test_x, test_y,
                                             [nn.metrics.BinaryCrossentropy, nn.metrics.BinaryAccuracy])
            end_time = time.time()
            results[idx]['elapsed_test'] += float(end_time - start_time)
            results[idx]['loss_test'] += loss_test
            results[idx]['acc_test'] += acc_test

    print()

    for i, result in enumerate(results):
        if len(models) > 1:
            print('Model #%d' % (i + 1))
        for key, val in result.items():
            print('%s: %f' % (key, val / num_run))
        print()


if __name__ == '__main__':
    test_models([task1.model, task2.model, task3.model],
                train_shape=(2, 1000),
                test_shape=(2, 100),
                lr=1, epochs=1000, num_run=100)
