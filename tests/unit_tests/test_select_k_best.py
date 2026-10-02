import os
from unittest import TestCase

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.feature_selection.select_k_best import SelectKBest


class TestSelectKBest(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)

    def test_fit_transform(self):
        skb = SelectKBest(k=2)
        skb.fit(self.dataset)
        new_dataset = skb.transform(self.dataset)

        self.assertTrue(skb.is_fitted())
        self.assertEqual(2, new_dataset.shape()[1])
        self.assertEqual(4, len(skb.F))
