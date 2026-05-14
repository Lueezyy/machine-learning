import numpy as np

class AdalineGD:
    '''ADAptive LInear NEuron classifier.'''
    def __init__(self, learning_rate=0.01, n_iterations=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X, y):
        '''Fit training data.'''
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias_ = np.float64(0.0)
        self.losses_ = []
        
        for i in range(self.n_iterations):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.weights_ += self.learning_rate * 2.0 * X.T.dot(errors) / X.shape[0]
            self.bias_ += self.learning_rate * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        '''Calculate net input.'''
        return np.dot(X, self.weights_) + self.bias_
    
    def activation(self, X):
        '''Compute linear activation.'''
        return X
    
    def predict(self, X):
        '''Return class label after unit step.'''
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)