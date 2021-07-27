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

class Layer:
    """
        This class is representing the connections between 2 layers of neurons
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward_pass(self, x):
        return np.dot(self.W * x) + self.b

