from ..Base import np
from ..Layer import Normalization

class BatchNorm(Normalization):
    def __init__(self):
        super().__init__()
        self.gamma = None
        self.beta = None

    def _init_weights_bias(self, col_size):
        self.gamma = self.gamma if self.gamma is not None else np.ones((1, col_size))
        self.beta = self.beta if self.beta is not None else np.zeros((1, col_size))

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.reshape((inputs.shape[0], -1))
        batch_size, inputs_size = inputs.shape
        self._init_weights_bias(inputs_size)

        if batch_size == 1:
            self.z_hat = inputs
        else:
            ### 批量均值
            mu = np.mean(inputs, axis=0, keepdims=True)
            ### 批量方差
            self.sigma = np.mean((inputs - mu)** 2, axis=0, keepdims=True) ** 0.5 + Normalization.CONST_E
            ### 數值依均值位移後 依方差為準值縮小
            self.z_hat = (inputs - mu) / self.sigma

        o = self.gamma * self.z_hat + self.beta if BatchNorm.is_training else self.z_hat
        if self.next is not None:
            self.next.forward(o.reshape(shape))

    def backward(self, deviation):
        shape = deviation.shape
        deviation = deviation.reshape((deviation.shape[0], -1))
        ### 更新
        self.gamma_increments = np.sum(deviation*self.z_hat, axis=0, keepdims=True)
        self.beta_increments = np.sum(deviation, axis=0, keepdims=True)

        if self.previous is not None:
            if deviation.shape[0] == 1:
                bk_deviation = deviation
            else:
                bk_deviation = self.sigma * (deviation \
                    - np.mean(deviation, axis=0, keepdims=True) \
                    - self.z_hat * np.mean(deviation * self.z_hat, axis=0, keepdims=True))
            self.previous.backward(bk_deviation.reshape(shape))


class LayerNorm(Normalization):
    def __init__(self):
        super().__init__()
        self.gamma = None
        self.beta = None

    def _init_weights_bias(self, col_size):
        self.gamma = self.gamma if self.gamma is not None else np.ones((col_size, 1))
        self.beta = self.beta if self.beta is not None else np.zeros((col_size, 1))

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.reshape((inputs.shape[0], -1))
        batch_size, inputs_size = inputs.shape
        self._init_weights_bias(batch_size)

        if inputs_size == 1:
            self.z_hat = inputs
        else:
            mu = np.mean(inputs, axis=1, keepdims=True)
            self.sigma = np.mean((inputs - mu)** 2, axis=1, keepdims=True) ** 0.5 + Normalization.CONST_E
            self.z_hat = (inputs - mu) / self.sigma

        o = self.gamma * self.z_hat + self.beta if LayerNorm.is_training else self.z_hat
        if self.next is not None:
            self.next.forward(o.reshape(shape))

    def backward(self, deviation):
        shape = deviation.shape
        deviation = deviation.reshape((deviation.shape[0], -1))
        ### 更新
        self.gamma_increments = np.sum(deviation*self.z_hat, axis=1, keepdims=True)
        self.beta_increments = np.sum(deviation, axis=1, keepdims=True)

        if self.previous is not None:
            if deviation.shape[1] == 1:
                bk_deviation = deviation
            else:
                bk_deviation = self.sigma * (deviation \
                    - np.mean(deviation, axis=1, keepdims=True) \
                    - self.z_hat * np.mean(deviation * self.z_hat, axis=1, keepdims=True))
            self.previous.backward(bk_deviation.reshape(shape))

