import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.metrics.mse import mse
from si.model_selection.split import train_test_split
from si.models.linear_regression import RidgeRegression
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares


class TestRidgeRegression(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), sep=",", features=True, label=True)
        self.train, self.test = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_mse(self):
        self.assertAlmostEqual(1.0, mse(np.array([1, 2, 3]), np.array([2, 3, 4])))

    def test_gradient_descent(self):
        model = RidgeRegression(l2_penalty=1, alpha=0.1, max_iter=2000, patience=100, scale=True)
        model.fit(self.train)
        self.assertEqual(6, len(model.theta))
        # cost must decrease along training
        self.assertLess(model.cost_history[max(model.cost_history)], model.cost_history[0])
        self.assertGreater(model.score(self.test), 0)

    def test_least_squares(self):
        model = RidgeRegressionLeastSquares(l2_penalty=1, scale=True)
        model.fit(self.train)
        self.assertEqual(6, len(model.theta))
        self.assertGreater(model.score(self.test), 0)

    def test_gd_matches_least_squares(self):
        gd = RidgeRegression(l2_penalty=1, alpha=0.1, max_iter=5000, patience=500, scale=True).fit(self.train)
        ls = RidgeRegressionLeastSquares(l2_penalty=1, scale=True).fit(self.train)
        np.testing.assert_allclose(gd.theta, ls.theta, atol=0.1)
        self.assertAlmostEqual(gd.theta_zero, ls.theta_zero, delta=0.1)
