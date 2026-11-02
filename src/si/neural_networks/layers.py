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

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self) -> tuple:
        return (self.n_units,)


class Dropout(Layer):
    """
    Dropout layer

    Parameters
    ----------
    probability: float
        Dropout rate, between 0 and 1

    Estimated parameters
    --------------------
    mask: numpy.ndarray
        Binomial mask used in the last forward pass
    input: numpy.ndarray
        Last input received
    output: numpy.ndarray
        Last output computed
    """

    def __init__(self, probability: float):
        self.probability = probability
        self._input_shape = None
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        self.input = input
        if not training:
            self.output = input
            return self.output

        scaling_factor = 1 / (1 - self.probability)
        self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
        self.output = input * self.mask * scaling_factor
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        return output_error * self.mask

    def output_shape(self) -> tuple:
        return self.input_shape()

    def parameters(self) -> int:
        return 0