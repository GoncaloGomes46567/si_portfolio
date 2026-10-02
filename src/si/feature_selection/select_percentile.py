from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Selects the top `percentile` % of features by score (f_classification by default)

    Parameters
    ----------
    score_func: Callable
        Function that receives a Dataset and returns (F, p)
    percentile: float
        Percentage of features to keep

    Estimated parameters
    ---------------------
    F: numpy.ndarray
    p: numpy.ndarray
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        n_features = len(self.F)
        n_select = int(round(n_features * self.percentile / 100))

        if n_select <= 0:
            return Dataset(dataset.X[:, :0], dataset.y, features=[], label=dataset.label)

        if n_select >= n_features:
            return Dataset(dataset.X.copy(), dataset.y, features=list(dataset.features), label=dataset.label)

        threshold = np.percentile(self.F, 100 - self.percentile)
        mask = self.F > threshold

        remaining = n_select - int(np.sum(mask))
        if remaining > 0:
            tie_idxs = np.where(self.F == threshold)[0]
            mask[tie_idxs[:remaining]] = True

        idxs = np.sort(np.where(mask)[0])
        new_X = dataset.X[:, idxs]
        new_features = np.array(dataset.features)[idxs].tolist()
        return Dataset(new_X, dataset.y, features=new_features, label=dataset.label)
