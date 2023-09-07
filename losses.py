import numpy as np

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    # dL/dy_pred = 2 * (y_pred - y_true) / n
    return 2 * (y_pred - y_true) / np.size(y_true)