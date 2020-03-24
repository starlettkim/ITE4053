import math
import random
from typing import List, Any

NUM_TRAIN = 1000
NUM_TEST = 100


class BinaryClassifier:

    def __init__(self):
        self.w1, self.w2, self.b = (0, 0, 0)

    @staticmethod
    def generate_data(size: int):
        datalist: List[{}] = []
        for i in range(size):
            data = {'x1': random.randint(-10, 10), 'x2': random.randint(-10, 10)}
            if data['x1'] + data['x2'] > 0:
                data['y'] = 1
            else:
                data['y'] = 0
            datalist.append(data)
        return datalist

    def train(self, datalist, learning_rate):
        pass

    def __sigmoid__(self, val):
        return 1 / (1 + math.exp(-val))

    def predict(self, x1, x2):
        pred_y = self.w1 * x1 + self.w2 * x2 + self.b
        pred_y = self.__sigmoid__(pred_y)
        return pred_y

    # Compute cross-entropy loss.
    def loss(self, datalist):
        y = 0
        for data in datalist:
            pred_y = self.predict(data['x1'], data['x2'])
            y -= data['y'] * math.log(pred_y) + (1 - data['y']) * math.log(1 - pred_y)
        y /= len(datalist)
        return y


classifier = BinaryClassifier()
train_data = classifier.generate_data(1000)
test_data = classifier.generate_data(100)

for iteration in range(101):
    if iteration:
        classifier.train(train_data, 1e-2)
    print('===== Iteration #' + str(iteration) + " =====")
    print('w1 = ' + str(classifier.w1) + ', w2 = ' + str(classifier.w2) + ', b = ' + str(classifier.b))
    print('train loss = ' + str(classifier.loss(train_data)))
    print('test loss = ' + str(classifier.loss(test_data)))
    print('train accuracy = ', end='')
    num_correct = 0
    for data in train_data:
        num_correct += (round(classifier.predict(data['x1'], data['x2'])) == data['y'])
    print(str(num_correct * 100 / len(train_data)) + '%')
    print('test accuracy = ', end='')
    num_correct = 0
    for data in test_data:
        num_correct += (round(classifier.predict(data['x1'], data['x2'])) == data['y'])
    print(str(num_correct * 100 / len(test_data)) + '%')
    print()