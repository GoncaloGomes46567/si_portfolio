import copy
from abc import ABCMeta, abstractmethod

import numpy as np


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__


class DenseLayer(Layer):
    """
    Fully connected layer

    Parameters
    ----------
    n_units: int
        Number of neurons
    input_shape: tuple
        Shape of the input, (n_features,) for tabular data

    Estimated parameters
    --------------------
    input: numpy.ndarray
        Last input received
    output: numpy.ndarray
        Last output computed
    weights: numpy.ndarray
        Weights, shape (n_features, n_units)
    biases: numpy.ndarray
        Biases, shape (1, n_units)
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer=None) -> 'DenseLayer':
        n_features = self.input_shape()[0]
        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))
        if optimizer is not None:
            self.w_opt = copy.deepcopy(optimizer)
            self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        return self.weights.size + self.biases.size

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def output_shape(self) -> tuple:
        return (self.n_units,)
