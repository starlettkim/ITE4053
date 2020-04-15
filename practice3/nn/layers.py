import numpy as np


class Layer(object):
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass

    def update(self):
        pass


class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)

        self.weights = np.random.randn(output_dim, input_dim) * .1
        self.bias = np.random.randn(output_dim, 1) * .1

    def forward(self, X: np.ndarray) -> np.ndarray:
        result = np.dot(self.weights, X) + self.bias
        return result

    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass

