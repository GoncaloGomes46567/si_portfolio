import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge regression solved with the closed-form (normal equation) solution

    Parameters
    ----------
    l2_penalty: float
        L2 regularization parameter
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
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        X = dataset.X.astype(float)
        y = dataset.y.astype(float)

        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1.0
            X = (X - self.mean) / self.std

        X = np.c_[np.ones(X.shape[0]), X]

        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])
        penalty_matrix[0, 0] = 0

        thetas = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(y)

        self.theta_zero = thetas[0]
        self.theta = thetas[1:]
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.X.astype(float)
        if self.scale:
            X = (X - self.mean) / self.std

        X = np.c_[np.ones(X.shape[0]), X]
        thetas = np.r_[self.theta_zero, self.theta]
        return X.dot(thetas)

    def _score(self, dataset: Dataset) -> float:
        y_pred = self._predict(dataset)
        return mse(dataset.y, y_pred)
