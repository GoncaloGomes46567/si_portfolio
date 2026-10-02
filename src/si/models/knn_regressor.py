from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.models.knn_classifier import euclidean_distance


class KNNRegressor(Model):
    """
    K-Nearest Neighbors regressor

    Parameters
    ----------
    k: int
        Number of neighbors to consider
    distance: Callable
        Distance function between a sample and the training dataset

    Estimated parameters
    ---------------------
    dataset: Dataset
        Training dataset
    """

    def __init__(self, k: int = 5, distance: Callable = euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray) -> float:
        distances = self.distance(sample, self.dataset.X)
        k_nearest_idxs = np.argsort(distances)[:self.k]
        k_nearest_values = self.dataset.y[k_nearest_idxs]
        return np.mean(k_nearest_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        predictions = self._predict(dataset)
        return rmse(dataset.y, predictions)
