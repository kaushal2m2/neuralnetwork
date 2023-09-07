from layer import Layer
import numpy as np

class Linear(layer):
    def __init__(self, input_size, neurons):
        # initialize weights and biases to random values
        self.weights = np.random.randn(neurons, input_size)
        self.biases = np.random.randn(neurons, 1)

    def forward(self, input):
        # save input for backward pass
        self.input = input

        # Y = W * X + b, return it to be passed as the input for the next layer
        self.output = np.dot(self.weights, input) + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # the output gradient is for our purposes dL/dy, we want to find the weights gradient
        # which is dL/dw = dL/dy * dy/dw
        # Y = W * X + b, so dy/dw = X
        # dL/dw = dL/dy * X, but we need the dimensions to be right, so we transpose X
        # the dimensions of dL/dy is (neurons, 1), and the dimensions of X.T is (1, input_size)
        # dL/dw = dL/dy * X.T
        weights_gradient = np.dot(output_gradient, self.input.T)

        # the input gradient is dL/dx = dL/dy * dy/dx
        # since each x (input layer neuron) is connected to each y (this layer neuron), there is a component for each
        # say wji is the weight for the jth neuron in the current layer from the ith neuron in the input layer
        # yj = sum (wji * xi) + bj, and dyj/dxi = wji
        # so dL/dxi = sum (dL/dyj * dyj/dxi) = sum (dL/dyj * wji)
        # this can be written as a matrix of wij * dL/dyj (note wij, so its transpose)
        input_gradient = np.dot(self.weights.T, output_gradient)

        #update weights by weight gradient, bias by output gradient
        # dL/dbj = dL/dyj * dyj/dbj = dL/dyj * 1 = dL/dyj
        # so the gradient for all the biases is just dL/dy, since they have the same dimensions
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient

        #to pass to the previous layer as its output layer
        return input_gradient
