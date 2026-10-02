import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.feature_selection.select_percentile import SelectPercentile


class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)

    def test_fit_transform(self):
        sp = SelectPercentile(percentile=50)
        sp.fit(self.dataset)
        new_dataset = sp.transform(self.dataset)

        self.assertTrue(sp.is_fitted())
        self.assertEqual(2, new_dataset.shape()[1])

    def test_tie_handling_example(self):
        # example from the slides: F-values with ties at the selection threshold
        F_values = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])

        def fake_score_func(dataset):
            return F_values, np.zeros_like(F_values)

        X = np.zeros((5, 10))
        y = np.array(['a', 'b', 'a', 'b', 'a'])
        dataset = Dataset(X, y, features=[f"f{i}" for i in range(10)], label='y')

        sp = SelectPercentile(score_func=fake_score_func, percentile=40)
        sp.fit(dataset)
        new_dataset = sp.transform(dataset)

        selected_idxs = [int(f[1:]) for f in new_dataset.features]
        self.assertEqual([3, 5, 6, 7], selected_idxs)
        self.assertEqual([5.6, 5.6, 7.8, 6.5], F_values[selected_idxs].tolist())
