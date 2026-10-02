import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split, stratified_train_test_split


class TestSplit(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)

    def test_train_test_split(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)

        self.assertEqual(120, train.shape()[0])
        self.assertEqual(30, test.shape()[0])
        self.assertEqual(150, train.shape()[0] + test.shape()[0])

    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)

        self.assertEqual(120, train.shape()[0])
        self.assertEqual(30, test.shape()[0])

        train_labels, train_counts = np.unique(train.y, return_counts=True)
        test_labels, test_counts = np.unique(test.y, return_counts=True)

        self.assertTrue(np.all(train_counts == 40))
        self.assertTrue(np.all(test_counts == 10))
