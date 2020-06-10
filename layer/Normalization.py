from ..Base import np
from ..Layer import Normalization

class BatchNorm(Normalization):

    def setAxis(self, inputs):
        return [0] if inputs.ndim == 2 else[0, 1, 2]


class LayerNorm(Normalization):

    def setAxis(self, inputs):
        return [1] if inputs.ndim == 2 else[1, 2, 3]


class InstanceNorm(Normalization):

    def setAxis(self, inputs):
        return [0, 1] if inputs.ndim == 2 else[0, 3]

