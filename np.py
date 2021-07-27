from abc import abstractmethod
import numpy as np

def create_weights_matrix(numRows, numCols):
    """Genreate normally distributed random numbers"""
    return np.random.default_rng().normal(
        loc=0,
        scale=1/(numRows*numCols),
        size=(numRows, numCols)
    )

class ActivationFunction:
    @abstractmethod
    def f(x):
        pass

    @abstractmethod
    def d(x):
        pass

class LeakyRelu(ActivationFunction):
    def __init__(self, alpha):
        self.alpha = alpha
    def f(self, x):
        return np.maximum(self.alpha*x, x)
    
    def d(self, x):
        return np.maximum(self.alpha, x > 0)

class LossFunction:
    @abstractmethod
    def loss(self, ouputs, targets):
        pass

    @abstractmethod
    def dloss(self, outputs, targets):
        pass

class MSE(LossFunction):
    def loss(self, outputs, targets):
        return np.mean(np.power(outputs - targets, 2))

    def dloss(self, outputs, targets):
        return 2 * (outputs - targets) / outputs.size


def create_bias_vector(length):
    return create_weights_matrix(length, 1)

class Layer:
    """
        This class is representing the connections between 2 layers of neurons
    """
    def __init__(self, inputs, outputs, act_func):
        self.W = create_weights_matrix(outputs, inputs)
        self.b = create_bias_vector(outputs)
        self.f = act_func.f

    def forward_pass(self, x):
        return self.f(np.dot(self.W, x) + self.b)

class Network:
    """Class representing a sequence of compatible layers"""

    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward_pass(out)
        return out
        

if __name__ == '__main__':
    layers = [
        Layer(3, 7, LeakyRelu(0.1)),
        Layer(7, 6, LeakyRelu(0.1)),
        Layer(6, 2, LeakyRelu(0.1))
    ]

    net = Network(layers)

    print(net.forward_pass(np.array([1, 2, 3]).reshape((3, 1))))
