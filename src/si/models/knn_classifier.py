from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Distance between a sample x and every sample in y
    """
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


class KNNClassifier(Model):
    """
    K-Nearest Neighbors classifier

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

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        distances = self.distance(sample, self.dataset.X)
        k_nearest_idxs = np.argsort(distances)[:self.k]
        k_nearest_labels = self.dataset.y[k_nearest_idxs]

        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        predictions = self._predict(dataset)
        return accuracy(dataset.y, predictions)
