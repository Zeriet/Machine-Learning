import numpy as np
from numpy.random import  seed

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier
    Paramaters
    ----------
    eta : float
        Learning rate (betweeen 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    ----------
    w_ : 1d - array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch
    shuffle: bool (defualt: True)
        Shuffles Traininf data every epcoh
        if True to prevent cylcles.
    random_state : int (default: None)
        Set random state for shuffling
        and initializing the weights.


    """
    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False

        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """Fit Training data

        Parameters
        ----------
        X : {array - like}, shape = [n_samples, n_features]
        Training vectors,
        where n_samples is the number of samples and n_features is the number of features
        Y: array-like, shape = [n_samples]
        Target values.

        Returns
        -------
        self: object
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range (self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in ZIP(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum (cost)/ len(y)
            self.cost_.append(avg_cost)

        return self

    def net_input(self, X):
        """calculate the input"""
        # print X[1:10]
        # print self.w_
        # print np.dot(X, self.w_[1:] + self.w_[0])[1:10]
        return np.dot(X, self.w_[1:] + self.w_[0])


    def activation(self, X):
        """Compute linear activations """
        return self.net_input(X)

    def predict(self, X):
        """Return class lable after unit step """
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def _initialize_weights(self, m):
        """Inittialize weights to zero"""
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline leatning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self._w_[1:] += self.eta * xi.dot(error)
        self._w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost














