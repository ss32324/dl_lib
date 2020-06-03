from .Base import Base, np

class Layer(Base):
    is_training = False
    def __init__(self):
        self.next: Layer = None
        self.previous: Layer = None


### Fully Connect
class FullyConnection(Layer):
    pass

### Activation Func
class Activation(Layer):
    def __init__(self):
        super().__init__()
        self.is_pair_with_lossfunc = False

### Normalization Func
class Normalization(Layer):
    pass

### Pool Layer
class Pool(Layer):
    pass

class Flatten(Layer):
    def forward(self, inputs):
        self.shape = inputs.shape
        self.next.forward(inputs.reshape(inputs.shape[0], -1))

    def backward(self, deviation):
        if self.previous is not None:
            self.previous.backward(deviation.reshape(self.shape))

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, inputs):
        if Layer.is_training:
            self.mask = (np.random.rand(*inputs.shape) > self.rate)
            o = self.mask * inputs
        else:
            o = inputs
        self.next.forward(o)

    def backward(self, deviation):
        if self.previous is not None:
            self.previous.backward(deviation * self.mask)
