import numpy as np

class Perceptron:
    '''Perceptron classifier.'''
    def __init__(self, learning_rate=0.01, n_iterations=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X, y):
        '''Fit training data.'''
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        print(self.weights_)
        self.bias_ = np.float64(0.0)
        self.errors_ = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                delta = self.learning_rate * (yi - self.predict(xi))
                self.weights_ += delta * xi
                self.bias_ += delta
                errors += int(delta != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        '''Calculate net input.'''
        return np.dot(X, self.weights_) + self.bias_
    
    def predict(self, X):
        '''Return class label after unit step.'''
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    