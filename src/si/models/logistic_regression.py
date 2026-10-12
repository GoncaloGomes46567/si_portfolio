import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function

class LogisticRegression(Model):
    """
    Logistic Regression model.
    """

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, penalty: str = None, C: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = penalty
        self.C = C
        self.weights = None
        self.bias = None

    def fit(self, dataset: Dataset) -> 'LogisticRegression':

        X, y = dataset.X, dataset.y
        n_samples, n_features = X.shape

        
        self.weights = np.zeros(n_features)
        self.bias = 0

        
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid_function(linear_model)

            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            
            if self.penalty == 'l2':
                dw += (self.C / n_samples) * self.weights  # L2 regularization

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self
