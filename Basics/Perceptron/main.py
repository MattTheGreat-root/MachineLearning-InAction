import numpy as np
from sklearn.datasets import make_blobs

X, Y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=41)


class F:
    @staticmethod
    def sign(x):
        if x < 0:
            return 0
        else:
            return 1


class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def forward(self, input):
        z = np.dot(self.weights, input) + self.bias
        y = F.sign(z)
        return y


class Optimizer:
    def __init__(self, model):
        self.model = model

    def update(self, x, y, y_hat):
        self.model.weights += (y - y_hat) * x
        self.model.bias += (y - y_hat)


input_size = 2
perceptron = Perceptron(input_size)

optimizer = Optimizer(perceptron)
n_iter = 1000
Y_hat = np.zeros(100)

for _ in range(n_iter):
    n = 0
    for x, y in zip(X, Y):
        y_hat = perceptron.forward(x)
        Y_hat[n] = y_hat
        n += 1
        optimizer.update(x, y, y_hat)

W1 = perceptron.weights[0]
W2 = perceptron.weights[1]
B = perceptron.bias

print(Y_hat, perceptron.weights, W1, W2, perceptron.bias, B)
