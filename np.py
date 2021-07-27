import numpy as np

def create_weights_matrix(numRows, numCols):
    """Genreate normally distributed random numbers"""
    return np.random.default_rng().normal(
        loc=0,
        scale=1/(numRows*numCols),
        size=(numRows, numCols)
    )

def create_bias_vector(length):
    return create_weights_matrix(length, 1)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha*x, x)

def mean_squared_erorr(outputs, targets):
    np.mean(np.power(outputs - targets, 2))

class Layer:
    """
        This class is representing the connections between 2 layers of neurons
    """
    def __init__(self, inputs, outputs, act_func):
        self.W = create_weights_matrix(outputs, inputs)
        self.b = create_bias_vector(outputs)
        self.f = act_func

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
        Layer(3, 7, leaky_relu),
        Layer(7, 6, leaky_relu),
        Layer(6, 2, leaky_relu)
    ]

    net = Network(layers)

    print(net.forward_pass(np.array([1, 2, 3]).reshape((3, 1))))
