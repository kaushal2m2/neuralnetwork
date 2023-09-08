import numpy as np
from scipy import signal
from layer import Layer

# input shape is (channels, height, width)
# the filter shape then is (channels, filter_height, filter_width)
# the first (depth) must be the same for both

# explanation for shapes
# say the input is a 4x4 image with 3 channels (for r,g,b -> each pixel has 3 values (channels))
# say there is only 1 filter, with a 2x2 shape. This filter must have 3 channels,
# since each input has 3 channels and it has to map something to each part of the input
# so you have a (3, 4, 4) shape for the input, and a (3, 2, 2) shape for the filter
# when we perform forward, imagine taking the filter over the pixels of the input
# we never travel in the channel dimension, only in the pixels, so the 3 channels get 
# mapped to 1 channel in the output. Since there is 1 filter, the output will only have 1 channel
# the output also will have a 2d shape of (3, 3), since there are 3 ways to place filter
# over each row. If we had multiple filters, there would be multiple layers of this output, each filter
# having its own layer. So, the output shape is (num_filters, 3, 3)
# from simple visualization, we can see that the last 2 dimensions are
# (input_height - filter_height + 1, input_width - filter_width + 1)

# takeaways
# the depth (channels) of the filter is the same as the depth (channels) of the input
# the output shape is (num_filters, input_height - filter_height + 1, input_width - filter_width + 1)

class Convolutional(Layer):
    def __init__(self, input_shape, filter_2d_shape, num_filters):
        c, h, w = input_shape
        fh, fw = filter_2d_shape # the depth of the filter is the same as input

        self.num_filters = num_filters
        self.input_shape = input_shape
        self.input_channels = c # number of channels per input (and kernal)

        self.output_shape = (num_filters, h-fh+1, w-fw+1)
        self.filter_shape = (num_filters, c, fh, fw) 
        # theres num_filters filters, each with (c, fh, fw)

        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)

        self.filters_momentum = np.zeros(self.filter_shape)
        self.biases_momentum = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)

        # to go forward on a convolutional layer, we need to overlap the filter with the input
        # and multiply and sum when the filter fully overlaps the input 
        # this is taken care of by signal.correlate2d, which, given 2 matrices performs this task
        # the mode is 'valid' since we only want to count when the filter is fully overlapping
        # so, for each filter, and for each channel in the input, we correlate the
        # jth channel of the input with the ith filter's jth channel, and
        # summing the values over all the input channels gets us the ouput for that filter
        for i in range(self.num_filters):
            for j in range(self.input_channels):
                self.output[i] += signal.correlate2d(self.input[j], self.filters[i,j], mode='valid')
        
        self.output += self.biases
        return self.output


    def backward(self, output_gradient, learning_rate):
        filters_gradient = np.zeros(self.filter_shape)
        input_gradient = np.zeros(self.input_shape)

        # we have to compute the gradients of the filter and the input
        # similar to the linear layer
        # we know that output gradient is dL/dy, and we want to find dL/df and dL/dx
        # we know that y = b + x (correlated with) f

        # for dL/df
        # in a simple example, 3x3 input and 2x2 filter, the output is a 2x2 matrix
        # y11 = b11 + x11*f11 + x12*f12 + x21*f21 + x22*f22
        # y12 = b12 + x12*f11 + x13*f12 + x22*f21 + x23*f22
        # etc..
        # dL/df11 = dL/dy11 * dy11/df11 = dL/dy11 * x11
        # the same applies for all, which works out to be X correlated with dL/dy

        # now for dL/dx
        # lets go piece by piece
        # dL/dx11 = dL/dy * dy/dx11 = dL/dy11 * k11
        # dL/dx12 = dL/dy * dy/dx12 = dL/dy11 * k12 + dL/dy12 * k11
        # dL/dx13 = dL/dy * dy/dx13 = dL/dy12 * k12
        # dL/dx21 = dL/dy * dy/dx21 = dL/dy11 * k21 + dL/dy21 * k11
        # this mismatching of the indices is the same as the convolution operation!
        # it's the convolution of the output gradient with the filter 
        # (remember that convolutions are the same as correlation, with the filter flipped 180)
        # this is done with 'full' because we also want to count when the filter isn't fully
        # in order to get the gradient for the input at channel j,
        # we sum up the convolutions of the output gradient at filter i, with the ith filter on channel j

        for i in range(self.num_filters):
            for j in range(self.input_channels):
                filters_gradient[i,j] = signal.correlate2d(self.input[j], output_gradient[i], mode='valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.filters[i,j], mode='full')

        # note: for biases the gradient is just the output gradient, for the same reasons as linear

        #momentum
        self.filters_momentum = 0.9 * self.filters_momentum - learning_rate * filters_gradient
        self.biases_momentum = 0.9 * self.biases_momentum - learning_rate * output_gradient
        self.filters += self.filters_momentum
        self.biases += self.biases_momentum

        return input_gradient
        
       
