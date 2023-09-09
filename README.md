# neuralnetwork

## About

This repo is my implementation of a basic **feedforward neural network with stochastic gradient descent** and a **convolutional neural network**, to help me learn the theory behind them. I consulted various research papers in order to learn and implement optimizations to make the neural net perform better. I've included comments in the code explaining where certain math comes from, and why it works. 

## Optimizations

### He initialization function

> ... leads to a zero-mean Gaussian distribution whose standard deviation (std) is sqrt(2/n<sub>l</sub>). This is our way of initialization. We also initialize b = 0. [1]

Due to this, I included code in initializations.py for He initialization, and used it to initializ the weights matrix in linear.py as a Gaussian Distribution with std sqrt(2/n), and set the biases to 0.

> initializations.py
```
def he(shape, input_size):
    return np.random.randn(*shape) * np.sqrt(2 / input_size)
```
> linear.py
```
self.weights = he((neurons, input_size), input_size)
self.biases = np.zeros((neurons, 1))
```


### Xavier initialization function

> _W ∼ U[- sqrt(6) / sqrt(n<sub>j</sub> + n<sub>j+1</sub>), sqrt(6) / sqrt(n<sub>j</sub> + n<sub>j+1</sub>)]_ [2]

This essentially says that the weights are initialized to a uniform distribution bounded by these values. _n<sub>j</sub>_ is the number of inputs, and _n<sub>j+1</sub>_ is the number of outputs. I included code in initializations.py for a xavier initialization, and included code in linear.py to use it.

> initializations.py
```
def xavier(shape, input_size, output_size):
    b = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-b, b, *shape)
```
> linear.py
```
self.weights = xavier((neurons, input_size), input_size, neurons)
self.biases = np.zeros((neurons, 1))
```

### Momentum with Stochastic Gradient Descent

> _v<sub>t+1</sub> = µv<sub>t</sub> - ε∇ f(⍬<sub>t</sub>) <br/> ⍬<sub>t+1</sub> = ⍬<sub>t</sub> + v<sub>t+1</sub>_ [3]

Where _v<sub>t</sub>_ is the how much we shift the weights in the _t_'th iteration, _⍬<sub>t</sub>_ is the weights, _∇ f_ is the gradient of the weights, _µ_ is the momentum coefficient, and _ε_ is the learning rate. 

I implemented this in linear.py by storing a momentum attribute for weights and biases, and updating it through every iteration, and using the stored attribute to update the weights and biases. I used 0.9 as the momentum coefficient. Similarly, momentum is implemented in convolutional.py for filters and biases.

```
self.weights_momentum = 0.9 * self.weights_momentum - learning_rate * weights_gradient
self.biases_momentum = 0.9 * self.biases_momentum - learning_rate * output_gradient

self.weights = self.weights + self.weights_momentum
self.biases = self.biases + self.biases_momentum
```

References: 

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034). [https://doi.org/10.48550/arXiv.1502.01852](https://doi.org/10.48550/arXiv.1502.01852)

[2] Glorot, X. & Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. _Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics_, in _Proceedings of Machine Learning Research_ 9:249-256 Available from [https://proceedings.mlr.press/v9/glorot10a.html](https://proceedings.mlr.press/v9/glorot10a.html).

[3] Sutskever, I., Martens, J., Dahl, G. & Hinton, G.. (2013). On the importance of initialization and momentum in deep learning. _Proceedings of the 30th International Conference on Machine Learning_, in _Proceedings of Machine Learning Research_ 28(3):1139-1147 Available from [https://proceedings.mlr.press/v28/sutskever13.html](https://proceedings.mlr.press/v28/sutskever13.html).