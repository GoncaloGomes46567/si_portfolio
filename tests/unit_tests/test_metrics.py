from unittest import TestCase

import numpy as np

from si.metrics.accuracy import accuracy
from si.metrics.rmse import rmse


class TestMetrics(TestCase):

    def test_accuracy(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])

        self.assertEqual(0.8, accuracy(y_true, y_pred))

    def test_rmse(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])

        self.assertEqual(0, rmse(y_true, y_pred))

        y_pred2 = np.array([2, 3, 4, 5])
        self.assertAlmostEqual(1.0, rmse(y_true, y_pred2))
