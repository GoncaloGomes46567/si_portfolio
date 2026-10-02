from typing import Tuple

import numpy as np
from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-way ANOVA F-test between each feature and the class labels
    Returns
    -------
    F: numpy.ndarray
    p: numpy.ndarray
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c, :] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p
