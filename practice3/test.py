import time
from typing import List, Dict
from tqdm import trange

import nn
from data import generate_data
import task1
import task2
import task3


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

    t = trange(num_run)
    t.set_description('Running tests on %d model(s)' % len(models))
    for _ in t:
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
    results = test_models([task1.model, task2.model, task3.model], 1, 1000, 100)
    for i, result in enumerate(results):
        print('model #%d' % (i + 1))
        for item in result.items():
            print('%s: %f' % item)
        print()
