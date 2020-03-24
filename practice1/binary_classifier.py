import random
from typing import List, Any


class BinaryClassifier:

    def __init__(self):
        self.w1, self.w2, self.b = (0, 0, 0)

    @staticmethod
    def generate_data(size: int):
        ret: List[{}] = []
        for i in range(size):
            data = {'x1': random.randint(-10, 10), 'x2': random.randint(-10, 10)}
            if data['x1'] + data['x2'] > 0:
                data['y'] = 1
            else:
                data['y'] = 0
            ret += data
        return ret

    def train(self, datalist, learning_rate):
        pass

    def predict(self, x1, x2):
        pass

    def loss(self):ㅎ
        pass


classifier = BinaryClassifier()

datalist = classifier.generate_data(1000)

for iteration in range(1, 101):
    classifier.train(datalist, 1e-2)
    print('Iteration #' + str(iteration) +
          ': w1 = ' + str(classifier.w1) + ', w2 = ' + str(classifier.w2) + ', b = ' + str(classifier.b))
