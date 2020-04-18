import time
from typing import List, Dict

import nn
from data import generate_data


def test_models(models: List[nn.Model],
                lr: float,
                epochs: int,
                num_run: int) \
        -> List[Dict[str, float]]:
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

    for _ in range(num_run):
        train_x, train_y = generate_data(-2, 2, (2, 1000))
        test_x, test_y = generate_data(-2, 2, (2, 100))

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

    for result in results:
        for key, val in result.items():
            result[key] = val / num_run

    return results


if __name__ == '__main__':
    models = [
        nn.models.Sequential([
            nn.layers.Dense(2, 1, nn.activations.Sigmoid)
        ]),
        nn.models.Sequential([
            nn.layers.Dense(2, 1, nn.activations.Sigmoid),
            nn.layers.Dense(1, 1, nn.activations.Sigmoid)
        ]),
        nn.models.Sequential([
            nn.layers.Dense(2, 3, nn.activations.Sigmoid),
            nn.layers.Dense(3, 1, nn.activations.Sigmoid)
        ])
    ]

    results = test_models(models, 1, 1000, 10)
    for i, result in enumerate(results):
        print('model #%d' % (i + 1))
        for item in result.items():
            print('%s: %f' % item)
        print()
