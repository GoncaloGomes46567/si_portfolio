import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    Linear model with L2 regularization, fit with gradient descent

    Parameters
    ----------
    l2_penalty: float
        L2 regularization parameter
    alpha: float
        Learning rate
    max_iter: int
        Maximum number of iterations
    patience: int
        Number of iterations without improvement before stopping
    scale: bool
        Whether to scale the data

    Estimated parameters
    ---------------------
    theta: numpy.ndarray
        Model coefficients
    theta_zero: float
        Y intercept
    mean, std: numpy.ndarray
        Used to scale the data
    cost_history: dict
        Cost value at each iteration
    """

    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001, max_iter: int = 2000,
                 patience: int = 100, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        X = dataset.X.astype(float)
        y = dataset.y.astype(float)

        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1.0
            X = self._scale(X)

        m, n = X.shape
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        best_cost = np.inf
        no_improvement = 0

        for i in range(self.max_iter):
            y_pred = X.dot(self.theta) + self.theta_zero
            error = y_pred - y

            gradient = (self.alpha / m) * error.dot(X)
            self.theta = self.theta * (1 - self.alpha * (self.l2_penalty / m)) - gradient
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(error)

            cost = self._cost_scaled(X, y)
            self.cost_history[i] = cost

            if cost < best_cost:
                best_cost = cost
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    break

        return self

    def _cost_scaled(self, X: np.ndarray, y: np.ndarray) -> float:
        m = X.shape[0]
        y_pred = X.dot(self.theta) + self.theta_zero
        return float((np.sum((y_pred - y) ** 2) + self.l2_penalty * np.sum(self.theta ** 2)) / (2 * m))

    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.X.astype(float)
        if self.scale:
            X = self._scale(X)
        return X.dot(self.theta) + self.theta_zero

    def _score(self, dataset: Dataset) -> float:
        y_pred = self._predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Cost function J between the predicted and real y values
        """
        X = dataset.X.astype(float)
        if self.scale:
            X = self._scale(X)
        return self._cost_scaled(X, dataset.y.astype(float))
