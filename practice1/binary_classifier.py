import math
import random
import time
from typing import List, Any

NUM_TRAIN = 1000
NUM_TEST = 100


class BinaryClassifier:

    def __init__(self):
        self.w1, self.w2, self.b = (0, 0, 0)

    @staticmethod
    def generate_data(size):
        datalist: List[{}] = []
        for i in range(size):
            data = {'x1': random.randint(-10, 10), 'x2': random.randint(-10, 10)}
            if data['x1'] + data['x2'] > 0:
                data['y'] = 1
            else:
                data['y'] = 0
            datalist.append(data)
        return datalist

    def __forward__(self, x1, x2):
        self.z = self.w1 * x1 + self.w2 * x2 + self.b
        self.a = 1 / (1 + math.exp(-self.z))
        MIN_VAL = 1e-10
        self.a = max(self.a, MIN_VAL)
        self.a = min(self.a, 1 - MIN_VAL)
        return self.a

    def __backward__(self, data):
        da = -data['y'] / self.a + (1 - data['y']) / (1 - self.a)
        dz = self.a * (1 - self.a) * da
        dw1 = data['x1'] * dz
        dw2 = data['x2'] * dz
        db = 1 * dz
        return dw1, dw2, db

    def train(self, datalist, learning_rate):
        batch_dw1, batch_dw2, batch_db = (0, 0, 0)
        for data in datalist:
            self.__forward__(data['x1'], data['x2'])
            dw1, dw2, db = self.__backward__(data)
            batch_dw1 += dw1 / len(datalist)
            batch_dw2 += dw2 / len(datalist)
            batch_db += db / len(datalist)

        self.w1 -= learning_rate * batch_dw1
        self.w2 -= learning_rate * batch_dw2
        self.b -= learning_rate * batch_db

    def predict(self, x1, x2):
        return round(self.__forward__(x1, x2))

    # Compute cross-entropy loss.
    def loss(self, datalist):
        batch_loss = 0
        for data in datalist:
            pred_y = self.__forward__(data['x1'], data['x2'])
            batch_loss -= data['y'] * math.log(pred_y) + (1 - data['y']) * math.log(1 - pred_y)
        batch_loss /= len(datalist)
        return batch_loss


def train_binary_classifier(num_train, num_test, num_iter, learn_rate):
    classifier = BinaryClassifier()
    train_data = classifier.generate_data(num_train)
    test_data = classifier.generate_data(num_test)
    for iteration in range(num_iter):
        classifier.train(train_data, learn_rate)
    ret = {'w1': classifier.w1, 'w2': classifier.w2, 'b': classifier.b, 'train_acc': 0, 'test_acc': 0}
    for data in train_data:
        ret['train_acc'] += (classifier.predict(data['x1'], data['x2']) == data['y']) / len(train_data)
    for data in test_data:
        ret['test_acc'] += (classifier.predict(data['x1'], data['x2']) == data['y']) / len(test_data)
    return ret


if __name__ == '__main__':
    classifier = BinaryClassifier()
    train_data = classifier.generate_data(NUM_TRAIN)
    test_data = classifier.generate_data(NUM_TEST)
    start = time.time()
    for iteration in range(1001):
        if iteration:
            classifier.train(train_data, 1e-2)
        print('===== Iteration #' + str(iteration) + " =====")
        print('w1 = ' + str(classifier.w1) + ', w2 = ' + str(classifier.w2) + ', b = ' + str(classifier.b))
        print('train loss = ' + str(classifier.loss(train_data)))
        print('test loss = ' + str(classifier.loss(test_data)))
        print('train accuracy = ', end='')
        num_correct = 0
        for data in train_data:
            num_correct += (classifier.predict(data['x1'], data['x2']) == data['y'])
        print(str(num_correct * 100 / len(train_data)) + '%')
        print('test accuracy = ', end='')
        num_correct = 0
        for data in test_data:
            num_correct += (classifier.predict(data['x1'], data['x2']) == data['y'])
        print(str(num_correct * 100 / len(test_data)) + '%')
        print()
    end = time.time()
    print('Time elapsed: ' + str(end - start) + 's')