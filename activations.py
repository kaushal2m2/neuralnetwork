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
    
    def backward(self, output_gradient):
        #dL/dxi = dL/dyi * dyi/dxi, dyi/dxi = activation_derivative(xi)
        #dL/dx = dL/dyi * activation_derivative(xi) for all i
        return np.multiply(output_gradient, self.activation_derivative(self.input))
    
class ReLU(Activation):
    def __init__(self):
        activation = lambda x: np.maximum(0, x)
        activation_derivative = lambda x: np.where(x > 0, 1, 0)
        super().__init__(activation, activation_derivative)

