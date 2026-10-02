import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ",", features: bool = False, label: bool = False) -> Dataset:
    """
    Reads a csv file and returns a Dataset object
    Returns
    -------
    Dataset
    """
    df = pd.read_csv(filename, sep=sep, header=0)

    if label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feature_names = list(df.columns[:-1]) if features else None
        label_name = df.columns[-1] if features else None
    else:
        X = df.to_numpy()
        y = None
        feature_names = list(df.columns) if features else None
        label_name = None

    return Dataset(X, y, features=feature_names, label=label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ",", features: bool = False, label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file
    """
    if features and dataset.features is not None:
        columns = list(dataset.features)
    else:
        columns = [f"feat_{i}" for i in range(dataset.X.shape[1])]

    data = pd.DataFrame(dataset.X, columns=columns)

    if label and dataset.has_label():
        label_name = dataset.label if (features and dataset.label is not None) else "y"
        data[label_name] = dataset.y

    data.to_csv(filename, sep=sep, index=False, header=True)
