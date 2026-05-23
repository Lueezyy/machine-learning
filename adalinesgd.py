import numpy as np

class AdalineGD:
    '''ADAptive LInear NEuron classifier.'''
    def __init__(self, learning_rate=0.01, n_iterations=50, shuffle=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        '''Fit training data.'''
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        
        for i in range(self.n_iterations):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self
    
    def partial_fit(self, X, y):
        '''Fit training data without reinitializing the weights.'''
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def initialize_weights(self, m):
        '''Initialize weights to small random numbers.'''
        self.rgen = np.random.RandomState(self.random_state)
        self.weights_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.bias_ = np.float64(0.0)
        self.weights_initialized = True

    def update_weights(self, xi, target):
        '''Apply Adaline learning rule to update the weights.'''
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.weights_ += self.learning_rate * 2.0 * xi * (error)
        self.bias_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss

    def net_input(self, X):
        '''Calculate net input.'''
        return np.dot(X, self.weights_) + self.bias_
    
    def activation(self, X):
        '''Compute linear activation.'''
        return X
    
    def predict(self, X):
        '''Return class label after unit step.'''
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)