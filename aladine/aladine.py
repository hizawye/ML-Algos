import numpy as np


class AladineGD:
    """
    Parameters
    -------------

    eta: int
        learning rate
    epochs: int
        passes over data
    random_state: int
        a seed to generate a random number for initialization of the weights

    Attributes
    -----------

    weights: 1D-array
        the weights after fitting
    bias: Scalar
        bias after fitting
    losses: list
        MSE(mean squared errors) loss function values in each epoch.

    """

    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Parameters
        -------------
        X: [n_examples, n_features]
        y: [n_examples]

        returns:
        self: obj
        """

        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, size=X.shape[1], scale=0.1)
        self.bias = np.float(0.0)

        self.losses = []

        for i in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.weights = self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.bias = self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.errors.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return X

    def predict(self, X, y):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
