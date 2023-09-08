import numpy as np
import random

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, loss_fn, loss_deriv, x_train, y_train, epochs = 1000, batch_size = None, learning_rate = 0.05):
        # if batch_size is None, then use all the data
        if batch_size is None:
            batch_size = len(x_train)
        
        for epoch in range(epochs):
            # get a random batch of data
            batch_indices = [i for i in range(len(x_train))]
            random.shuffle(batch_indices)
            idx = 0 if batch_size == len(x_train) else np.random.default_rng().integers(len(x_train)-batch_size)
            batch_indices = batch_indices[idx:idx+batch_size]
            x_batch = [x_train[i] for i in batch_indices]
            y_batch = [y_train[i] for i in batch_indices]
            
            # train on this batch
            for x,y in zip(x_batch, y_batch):
                # forward pass
                output = self.predict(x)
                
                # backward pass
                loss_gradient = loss_deriv(y, output)
                for layer in reversed(self.layers):
                    loss_gradient = layer.backward(loss_gradient, learning_rate)
            
            # print progress
            predictions = [self.predict(x) for x in x_batch]
            loss = 0
            for i in range(batch_size):
                loss += loss_fn(y_batch[i], predictions[i])
            loss = loss / batch_size
            print("Epoch %d/%d loss: %.3f" % (epoch+1, epochs, loss))