from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
    
    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input) #pass into activation function for output
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # dL/dxi = dL/dyi * dyi/dxi, dyi/dxi = activation_derivative(xi)
        # dL/dx = dL/dyi * activation_derivative(xi) for all i
        # hadamard product since we just want to multiply the terms in the same positions
        return np.multiply(output_gradient, self.activation_derivative(self.input))
    
class ReLU(Activation):
    def __init__(self):
        activation = lambda x: np.maximum(0, x)
        activation_derivative = lambda x: np.where(x > 0, 1, 0)
        super().__init__(activation, activation_derivative)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x : 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x : sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        curr = np.exp(input)
        self.output = curr / np.sum(curr)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)