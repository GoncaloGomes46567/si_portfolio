import numpy as np

from si.data.dataset import Dataset


def read_data_file(filename: str, sep: str = ",", label: bool = False) -> Dataset:
    """
    Reads a data file (no header) and returns a Dataset object
    Returns
    -------
    Dataset
    """
    data = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None

    return Dataset(X, y)


def write_data_file(filename: str, dataset: Dataset, sep: str = ",", label: bool = False) -> None:
    """
    Writes a Dataset object to a data file (no header)
    """
    if label and dataset.has_label():
        y = dataset.y.reshape(-1, 1)
        data = np.hstack((dataset.X, y))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)
