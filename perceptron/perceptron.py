import numpy as np


class Perceptron:
    """
    Paramaters
    -----------
    epochs : int
        number of iterations
    eta : int
        learning rate
    randome state : int
        seed for random number generator

    Attributes
    -----------
    weights_ : array
    bias_ : int
    errors_ : list
        number of misclassification in each epoch
    """

    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, size=X.shape[1], scale=0.01)
        self.bias_ = np.float_(0.0)
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for xi, targets in zip(X, y):
                update = self.eta * (targets - self.predict(xi))
                self.weights_ += update * xi
                self.bias_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            # print(self.weights_)
        return self

    def net_input(self, X):
        "return z in for sigma(z)= wx + b"
        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
