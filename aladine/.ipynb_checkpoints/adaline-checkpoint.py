import numpy as np



class AladineGD:
    """
    ADAptive LInear NEuron classifier

    Parameters
    ----------
    eta : float
        Learning rate
    epochs : int
        Passes over the training data
    random_state : int
        Random seed for reproducible weight initialization

    Attributes
    ----------
    weights : ndarray
        Model weights after training
    bias : float
        Bias term after training
    losses : list
        Mean squared error for each epoch
    """

    def __init__(self, eta=0.01, epochs=10, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = 0.0
        self.losses = []

        for _ in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.weights += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.bias += self.eta * 2.0 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
