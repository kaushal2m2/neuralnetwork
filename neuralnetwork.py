import numpy as np

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
            batch_indices = np.random.Generator.integers(len(x_train), size = batch_size)
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            x_batch = np.reshape(x_batch, (-1, batch_size))
            y_batch = np.reshape(y_batch, (-1, batch_size))
            
            # train on this batch
            error = 0
            for x,y in zip(x_batch, y_batch):
                # forward pass
                output = self.predict(x)
                
                # backward pass
                loss_gradient = loss_deriv(y, output)
                for layer in reversed(self.layers):
                    loss_gradient = layer.backward(loss_gradient, learning_rate)
            
            # print progress
            loss = loss_fn(y_batch, self.predict(x_batch))
            print("Epoch %d/%d loss: %.3f" % (epoch, epochs, loss))