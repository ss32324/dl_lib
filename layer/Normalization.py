from ..Base import np
from ..Layer import Normalization

class BatchNorm(Normalization):

    def forward(self, inputs):
        self._init_weights_bias(shape=(1, inputs.shape[-1]))
        axis = (0) if len(inputs.shape) == 2 else(0, 1, 2)
        if BatchNorm.is_training:
            ### 批量均值
            self.mu = np.mean(inputs, axis=axis, keepdims=True)
            ### 批量方差
            self.sigma = np.mean((inputs - self.mu)** 2, axis=axis, keepdims=True) ** 0.5 + Normalization.CONST_E
        ### 數值依均值位移後 依方差為準值縮小
        self.z_hat = (inputs - self.mu) / self.sigma
        o = self.gamma * self.z_hat + self.beta
        if self.next is not None:
            self.next.forward(o)

    def backward(self, deviation):
        axis = (0) if len(deviation.shape) == 2 else (0, 1, 2)
        ### 更新
        self.gamma_increments = np.sum(deviation*self.z_hat, axis=axis)
        self.beta_increments = np.sum(deviation, axis=axis)

        if self.previous is not None:
            bk_deviation = self.sigma * (deviation \
                - np.mean(deviation, axis=axis, keepdims=True) \
                - self.z_hat * np.mean(deviation * self.z_hat, axis=axis, keepdims=True))
            self.previous.backward(bk_deviation)


class LayerNorm(Normalization):

    def forward(self, inputs):
        self._init_weights_bias(shape=(inputs.shape[0], 1))
        shape = inputs.shape
        inputs = inputs.reshape(shape[0], -1)
        ### 批量均值
        mu = np.mean(inputs, axis=1, keepdims=True)
        ### 批量方差
        self.sigma = np.mean((inputs - mu)** 2, axis=1, keepdims=True) ** 0.5 + Normalization.CONST_E
        ### 數值依均值位移後 依方差為準值縮小
        self.z_hat = inputs if (inputs == mu).any() else(inputs - mu) / self.sigma

        o = self.gamma * self.z_hat.reshape(inputs.shape[0], -1) + self.beta if LayerNorm.is_training else self.z_hat
        if self.next is not None:
            self.next.forward(o.reshape(shape))

    def backward(self, deviation):
        shape = deviation.shape
        deviation = deviation.reshape(shape[0], -1)
        ### 更新
        self.gamma_increments = np.sum(deviation * self.z_hat, axis=1, keepdims=True)
        self.beta_increments = np.sum(deviation, axis=1, keepdims=True)

        if self.previous is not None:
            bk_deviation = self.sigma * (deviation \
                - np.mean(deviation, axis=1, keepdims=True) \
                - self.z_hat * np.mean(deviation * self.z_hat, axis=1, keepdims=True))
            self.previous.backward(bk_deviation.reshape(shape))

