# neuralnetwork

## About

This repo is my implementation of a basic **feedforward neural network with stochastic gradient descent**, to help me learn the theory behind them. I consulted various research papers in order to learn and implement optimizations to make the neural net perform better. I've included comments explaining where certain math comes from, and why it works. 

## Optimizations

### He initialization function

> ... leads to a zero-mean Gaussian distribution whose standard deviation (std) is sqrt(2/n<sub>l</sub>). This is our way of initialization. We also initialize b = 0.

Due to this, I initialized the weights matrix in linear.py as a Gaussian Distribution with std sqrt(2/n), and set the biases to 0.

```
self.weights = np.random.randn(neurons, input_size) * np.sqrt(2 / input_size)
self.biases = np.zeros((neurons, 1))
```

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034). [https://doi.org/10.48550/arXiv.1502.01852](https://doi.org/10.48550/arXiv.1502.01852)

### Momentum with Stochastic Gradient Descent

> * *v<sub>t+1</sub> = µv<sub>t</sub> - ε∇f(⍬<sub>t</sub>) \n ⍬<sub>t+1</sub> = ⍬<sub>t</sub> + v<sub>t+1</sub>* *

Where v<sub>t</sub> is the weights gradient in the tth iteration, and µ is the momentum coefficient. 

I implemented this in linear.py by storing a momentum attribute for weights and biases, and updating it through every iteration, and using the stored attribute to update the weights and biases. I used 0.9 as the momentum coefficient, so 0.1 for ε.


```
self.weights_momentum = 0.9 * self.weights_momentum - 0.1 * weights_gradient
self.biases_momentum = 0.9 * self.biases_momentum - 0.1 * output_gradient

self.weights = self.weights + learning_rate * self.weights_momentum
self.biases = self.biases + learning_rate * self.biases_momentum
```

[2] Sutskever, I., Martens, J., Dahl, G. & Hinton, G.. (2013). On the importance of initialization and momentum in deep learning. * *Proceedings of the 30th International Conference on Machine Learning* *, in * *Proceedings of Machine Learning Research* * 28(3):1139-1147 Available from [https://proceedings.mlr.press/v28/sutskever13.html](https://proceedings.mlr.press/v28/sutskever13.html).



