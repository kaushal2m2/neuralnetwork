class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self):
        # to be overridden, returns output to be passed to next layer
        pass

    def backward(self):
        # to be overridden, returns gradient to be passed to previous layer
        pass
