import numpy as np

class AdalineGD:
    '''ADAptive LInear NEuron classifier.'''
    def __init__(self, learning_rate=0.01, n_iterations=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias_ = np.float64(0.0)
        self.losses_ = []
        return self
    
    def net_input(self, X):
        '''Calculate net input.'''
        return np.dot(X, self.weights_) + self.bias_