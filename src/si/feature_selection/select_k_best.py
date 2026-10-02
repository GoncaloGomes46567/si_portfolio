from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Selects the k features with the highest score (f_classification by default)

    Parameters
    ----------
    score_func: Callable
        Function that receives a Dataset and returns (F, p)
    k: int
        Number of features to select

    Estimated parameters
    ---------------------
    F: numpy.ndarray
    p: numpy.ndarray
    """

    def __init__(self, score_func: Callable = f_classification, k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        idxs = np.argsort(self.F)[-self.k:]
        idxs = np.sort(idxs)

        new_X = dataset.X[:, idxs]
        new_features = np.array(dataset.features)[idxs].tolist()
        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)
