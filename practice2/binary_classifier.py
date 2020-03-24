import numpy as np
import time


DIM_X = 2           # Dimension of data
NUM_TRAIN = 1000    # Number of train data
NUM_TEST = 100      # Number of test data
LEARN_RATE = 1e-2   # Learning rate
NUM_ITER = 100      # Number of iterations


class BinaryClassifier:

    def __init__(self):
        self.w = np.zeros((1, DIM_X))   # (1, D)
        self.b = 0

    @staticmethod
    def generate_data(size):
        X = 20 * np.random.rand(size, DIM_X) - 10   # (N, D)
        y = (np.sum(X, 1) > 0).astype(int).reshape(size, 1)   # (N, 1)
        return X, y

    def __forward__(self, X):
        self.z = np.dot(X, self.w.T) + self.b   # (N, 1)
        self.a = 1 / (1 + np.exp(-self.z))   # (N, 1)
        return self.a

    def __backward__(self, X, y):
        da = -y / self.a + (1 - y) / (1 - self.a)   # (N, 1)
        dz = self.a * (1 - self.a) * da     # (N, 1)
        print(self.a.shape)
        dw = X * dz     # (N, D)
        db = 1 * dz     # (N, 1)
        return dw, db

    def train(self, X, y, learning_rate):
        self.__forward__(X)
        dw, db = self.__backward__(X, y)
        dw = np.mean(dw, 0)  # (1, D)
        db = np.mean(db, 0)  # (1, 1)
        self.w -= learning_rate * dw
        self.b -= learning_rate * db


classifier = BinaryClassifier()
train_X, train_y = classifier.generate_data(NUM_TRAIN)
test_X, test_y = classifier.generate_data(NUM_TEST)

start = time.time()

for iteration in range(NUM_ITER + 1):
    if iteration:
        classifier.train(train_X, train_y, 1e-2)
    print('===== Iteration #' + str(iteration) + " =====")
    for i in range(classifier.w.shape[1]):
        print('w' + str(i + 1) + ' = ' + str(classifier.w[0][i]))
    print('b = ' + str(classifier.b))

end = time.time()
print('Time elapsed: ' + str(end - start) + 's')