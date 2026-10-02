import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of correctly classified samples
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)
