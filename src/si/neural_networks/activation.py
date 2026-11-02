from abc import abstractmethod
import numpy as np
from si.neural_networks.layers import Layer
 
 
class ActivationLayer(Layer):
 
    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        self.input = input
        self.output = self.activation_function(input)
        return self.output
 
    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        return self.derivative(self.input) * output_error
 
    @abstractmethod
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError
 
    @abstractmethod
    def derivative(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError
 
    def output_shape(self) -> tuple:
        return self.input_shape()
 
    def parameters(self) -> int:
        return 0
 
 
class ReLUActivation(ActivationLayer):
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        return np.where(input >= 0, 1, 0)
 
 
class SigmoidActivation(ActivationLayer):
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        f = self.activation_function(input)
        return f * (1 - f)
 
 
class TanhActivation(ActivationLayer):
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(input) ** 2
 
 
class SoftmaxActivation(ActivationLayer):
 
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        exp = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
 
    def derivative(self, input: np.ndarray) -> np.ndarray:
        f = self.activation_function(input)
        return f * (1 - f)
 