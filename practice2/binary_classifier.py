import numpy as np
import time

DIM_X = 2  # Dimension of data


class BinaryClassifier:

    def __init__(self):
        self.w = np.zeros((1, DIM_X), dtype=np.float64)  # (1,D)
        self.b = 0

    @staticmethod
    def generate_data(size):
        X = np.random.randint(-10, 11, (size, DIM_X))   # (N,D)
        y = (np.sum(X, 1) > 0).astype(int)  # (N,)
        return X, y

    def __forward__(self, X):
        self.z = np.dot(self.w, X.T) + self.b  # (1,N)
        self.a = 1 / (1 + np.exp(-self.z))  # (1,N)
        MIN_MARGIN = 2 ** -53
        self.a = np.maximum(MIN_MARGIN, np.minimum(1 - MIN_MARGIN, self.a))
        return self.a

    def __backward__(self, X, y):
        da = -y / self.a + (1 - y) / (1 - self.a)  # (1,N)
        dz = self.a * (1 - self.a) * da  # (1,N)
        dw = np.mean(X * dz.T, 0)  # (D,)
        db = np.mean(1 * dz)
        return dw, db

    def train(self, X, y, learning_rate):
        self.__forward__(X)
        dw, db = self.__backward__(X, y)
        self.w -= learning_rate * dw
        self.b -= learning_rate * db

    def predict(self, X):
        return np.round(self.__forward__(X))

    def loss(self, X, y):
        pred_y = self.__forward__(X)
        return -np.mean(y * np.log(pred_y) + (1 - y) * np.log(1 - pred_y))


def train_binary_classifier(num_train, num_test, num_iter, learn_rate):
    classifier = BinaryClassifier()
    train_X, train_y = classifier.generate_data(num_train)
    test_X, test_y = classifier.generate_data(num_test)
    for iteration in range(num_iter):
        classifier.train(train_X, train_y, learn_rate)
    return {'w': classifier.w, 'b': classifier.b,
            'train_loss': classifier.loss(train_X, train_y),
            'test_loss': classifier.loss(test_X, test_y),
            'train_acc': 100 * np.mean(classifier.predict(train_X) == train_y),
            'test_acc': 100 * np.mean(classifier.predict(test_X) == test_y)}


if __name__ == '__main__':
    NUM_TRAIN = 1000  # Number of train data
    NUM_TEST = 100  # Number of test data
    LEARN_RATE = 1e-2  # Learning rate
    NUM_ITER = 1000  # Number of iterations

    classifier = BinaryClassifier()
    train_X, train_y = classifier.generate_data(NUM_TRAIN)
    test_X, test_y = classifier.generate_data(NUM_TEST)
    start = time.time()
    for iteration in range(NUM_ITER + 1):
        if iteration:
            classifier.train(train_X, train_y, LEARN_RATE)
        print('===== Iteration #' + str(iteration) + " =====")
        for i in range(classifier.w.shape[1]):
            print('w' + str(i + 1) + ' = ' + str(classifier.w[0][i]))
        print('b = ' + str(classifier.b))
        print('Train loss = ' + str(classifier.loss(train_X, train_y)))
        print('Test loss = ' + str(classifier.loss(test_X, test_y)))
        print('Train accuracy = ' + str(100 * np.mean(classifier.predict(train_X) == train_y)) + '%')
        print('Test accuracy = ' + str(100 * np.mean(classifier.predict(test_X) == test_y)) + '%')
        print()
    end = time.time()
    print('Time elapsed: ' + str(end - start) + 's')
