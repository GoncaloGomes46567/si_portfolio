import os
from unittest import TestCase

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.feature_selection.variance_threshold import VarianceThreshold


class TestVarianceThreshold(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)

    def test_fit_transform(self):
        vt = VarianceThreshold(threshold=0.5)
        vt.fit(self.dataset)
        new_dataset = vt.transform(self.dataset)

        self.assertTrue(vt.is_fitted())
        self.assertTrue(new_dataset.shape()[1] < self.dataset.shape()[1])
        self.assertTrue(all(f in self.dataset.features for f in new_dataset.features))
