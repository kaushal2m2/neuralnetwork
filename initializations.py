import numpy as np

def he(shape, input_size):
    return np.random.randn(*shape) * np.sqrt(2 / input_size)

def xavier(shape, input_size, output_size):
    b = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-b, b, *shape)