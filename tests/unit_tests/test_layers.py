from unittest import TestCase

import numpy as np

from si.neural_networks.layers import DenseLayer


class TestDenseLayer(TestCase):

    def setUp(self):
        self.layer = DenseLayer(4, (3,)).initialize()

    def test_initialize(self):
        self.assertEqual(self.layer.weights.shape, (3, 4))
        self.assertEqual(self.layer.biases.shape, (1, 4))
        self.assertTrue(np.all(self.layer.weights >= -0.5))
        self.assertTrue(np.all(self.layer.weights < 0.5))
        self.assertTrue(np.all(self.layer.biases == 0))

    def test_forward_propagation(self):
        out = self.layer.forward_propagation(np.random.rand(10, 3))
        self.assertEqual(out.shape, (10, 4))

    def test_parameters(self):
        self.assertEqual(self.layer.parameters(), 3 * 4 + 4)

    def test_output_shape(self):
        self.assertEqual(self.layer.output_shape(), (4,))

    def test_layer_name(self):
        self.assertEqual(self.layer.layer_name(), "DenseLayer")

    def test_forward_known_values(self):
        layer = DenseLayer(3, (2,)).initialize()
        layer.weights = np.array([[0.1, 0.2, -0.1],
                                  [0.3, -0.2, 0.4]])
        layer.biases = np.array([[0.1, 0, -0.1]])
        out = layer.forward_propagation(np.array([[1, 2]]))
        np.testing.assert_allclose(out, [[0.8, -0.2, 0.6]])
