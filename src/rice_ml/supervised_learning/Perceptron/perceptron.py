import numpy as np

class Perceptron(object):
    def __init__(self, eta = .5, epochs=50):
        self.eta = eta
        self.epochs = epochs
        
    def train(self, X, y):
        self.w_ = np.random.rand(1 + X.shape[1])  # Initialize weights randomly
        self.errors_ = []  # Track errors for each epoch

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))  # Correct update rule
                self.w_[:-1] += update * xi  # Update weights
                self.w_[-1] += update  # Update bias
                errors += int(update != 0)  # Count misclassifications
            self.errors_.append(errors)  # Append errors for this epoch
            if errors == 0:  # Stop early if no errors
                break

        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[:-1]) + self.w_[-1]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)