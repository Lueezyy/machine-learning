import numpy as np

class LogisticRegressionGD:
    '''Gradient descent-based logistic regression classifier.'''
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        '''Fit training data.'''
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
        return self

    def net_input(self, X):
        '''Calculate net input.'''
        return np.dot(X, self.w_) + self.b_