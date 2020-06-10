from ..Base import np
from ..Layer import Activation

class Sigmoid(Activation):

    def act(self, x):
        # self.outputs = 1 / np.exp(-x)
        e = np.exp(x)
        y = e / (e + 1)
        return super().act(y)

    def dact(self, y):
        dy = y * (1 - y)
        return super().dact(dy)


class ReLU(Activation):

    def act(self, x):
        y = np.where(x > 0, x, 0)
        return super().act(y)

    def dact(self, y):
        dy = np.where(y > 0, 1, 0)
        return super().dact(dy)


class Softmax(Activation):

    def act(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        y = e / np.sum(e, axis=1, keepdims=True)
        return super().act(y)

    def dact(self, y):
        dy = y * (1 - y)
        return super().dact(dy)

