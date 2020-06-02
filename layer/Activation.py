from ..Base import np
from ..Layer import Activation

class Sigmoid(Activation):

    def forward(self, x):
        # self.outputs = 1 / np.exp(-x)
        e = np.exp(x)
        self.outputs = e / (e + 1)
        if self.next is not None:
            self.next.forward(self.outputs)

    def backward(self, deviation):
        if self.previous is not None:
            if not self.is_pair_with_lossfunc:
                bk_deviation = deviation * (self.outputs * (1 - self.outputs))
                self.previous.backward(bk_deviation)
            else:
                self.previous.backward(deviation)

class ReLU(Activation):

    def forward(self, x):
        self.outputs = np.where(x > 0, x, 0)
        if self.next is not None:
            self.next.forward(self.outputs)

    def backward(self, deviation):
        if self.previous is not None:
            if not self.is_pair_with_lossfunc:
                bk_deviation = deviation * np.where(self.outputs > 0, 1, 0)
                self.previous.backward(bk_deviation)
            else:
                self.previous.backward(deviation)

class Softmax(Activation):

    def forward(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.outputs = e / np.sum(e, axis=1, keepdims=True)
        if self.next is not None:
            self.next.forward(self.outputs)

    def backward(self, deviation):
        if self.previous is not None:
            if not self.is_pair_with_lossfunc:
                bk_deviation = deviation * (self.outputs * (1 - self.outputs))
                self.previous.backward(bk_deviation)
            else:
                self.previous.backward(deviation)
