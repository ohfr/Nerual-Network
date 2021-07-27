from abc import abstractmethod
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
    
    def df(self, x):
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

class Layer:
    """
        This class is representing the connections between 2 layers of neurons
    """
    def __init__(self, inputs, outputs, act_func):
        self.W = create_weights_matrix(outputs, inputs)
        self.b = create_bias_vector(outputs)
        self.f = act_func.f
        self.act_func = act_func

    def forward_pass(self, x):
        return self.f(np.dot(self.W, x) + self.b)

class Network:
    """Class representing a sequence of compatible layers"""

    def __init__(self, layers, lr, loss_func):
        self.layers = layers
        self.lr = lr
        self.loss_func = loss_func

    def forward_pass(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward_pass(out)
        return out
    
    def train(self, x, target):
        """Train the network on the input x and the target value"""
        # accumulate all intermediate outputs
        xs = [x]

        for layer in self.layers:
            xs.append(layer.forward_pass(xs[-1]))

        dx = self.loss_func.dloss(xs.pop(), target)

        # zip allows to loop 2 iterables at once
        for layer, x in zip(self.layers[::-1], xs[::-1]):
            db = dx * layer.act_func.df(np.dot(layer.W, x) + layer.b)
            dx = np.dot(layer.W.T, db)
            dw = np.dot(db, x.T)
            layer.W -= self.lr * dw
            layer.b -= self.lr * db

    def loss(self, x, target):
        return self.loss_func.loss(self.forward_pass(x), target)
        

if __name__ == '__main__':
    layers = [
        Layer(3, 7, LeakyRelu(0.1)),
        Layer(7, 6, LeakyRelu(0.1)),
        Layer(6, 1, LeakyRelu(0.1))
    ]

    target = np.array([[0]])

    net = Network(layers, 0.001, MSE())

    inputs = [
        create_bias_vector(3) for _ in range(1000)
    ]
    # Test network before training it
    loss = 0
    for input in inputs:
        loss += net.loss(input, target)
    print(loss)

    for _ in range(10000):
        x = create_bias_vector(3)
        net.train(x, target)
        
    loss = 0
    for input in inputs:
        loss += net.loss(input, target)
    print(loss)