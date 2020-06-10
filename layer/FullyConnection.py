from ..Base import np
from ..Layer import FullyConnection


class Dense(FullyConnection):
    def __init__(self, no_of_outputs, weight_scale=1, bias_scale=1):
        super().__init__()
        self.no_of_outputs = no_of_outputs
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.weights = None
        self.bias = None

    def __init_weight_bias(self, no_of_inputs):
        ### ref. https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        if self.weights is None:
            self.weights = np.random.randn(no_of_inputs, self.no_of_outputs) * ((2 / (no_of_inputs + self.no_of_outputs))** 0.5) * self.weight_scale
            # self.weights = np.random.uniform(-1, 1, (no_of_inputs, self.no_of_outputs)) * self.weight_scale
        if self.bias is None:
            self.bias = np.ones((1, self.no_of_outputs)) * self.bias_scale

    def forward(self, inputs):
        self.inputs = np.asarray(inputs)
        self.__init_weight_bias(inputs.shape[1])
        if self.next is not None:
            # WX
            self.next.forward(np.dot(self.inputs, self.weights) + self.bias)

    def backward(self, deviation):
        # 輸出對權重的增量
        self.weights_increments = np.dot(self.inputs.T, deviation) / deviation.shape[0]
        self.bias_increments = np.sum(deviation, axis=0, keepdims=True) / deviation.shape[0]
        if self.previous is not None:
            # 往前一層的偏差
            self.previous.backward(np.dot(deviation, self.weights.T))


class Conv(FullyConnection):
    ### ref. http://deeplearning.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/
    def __init__(self, k_no, k_size, strides=1, is_add_padding=False, weight_scale=1, bias_scale=1):
        super().__init__()
        self.k_no = k_no
        self.k_size = k_size
        self.strides = strides
        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.is_add_padding = is_add_padding
        self.weights = None
        self.bias = None

    def __init_weight_bias(self, k_channel):
        if self.weights is None:
            self.weights = np.random.randn(self.k_no, self.k_size, self.k_size, k_channel) * ((2 / (self.k_no + self.k_size + self.k_size + k_channel))** 0.5) * self.weight_scale
            # self.weights = np.random.uniform(-1, 1, (self.k_no, self.k_size, self.k_size, k_channel)) * self.weight_scale
        if self.bias is None:
            self.bias = np.ones((self.k_no, 1)) * self.bias_scale

    def split_feature_map(x, k_size, strides, p_width=None):
        ### ref. https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman
        # 加入邊界
        if p_width is not None:
            x = np.pad(x, mode='constant', pad_width=((0, 0), (p_width, p_width), (p_width, p_width), (0, 0)))
        # 計算輸出形狀
        o_h = (x.shape[1] - k_size) // strides + 1
        o_w = (x.shape[2] - k_size) // strides + 1
        o_shape = (x.shape[0], o_h, o_w, k_size, k_size, x.shape[-1])
        o_strides = (x.strides[0], x.strides[1]*strides, x.strides[2]*strides, *x.strides[1:])
        return np.lib.stride_tricks.as_strided(x, o_shape, o_strides)

    def forward(self, inputs):
        self.__init_weight_bias(inputs.shape[-1])
        # 以kernel_size及strides對圖片進行切割
        pad_w = (self.k_size - 1) // 2
        self.split_imgs = Conv.split_feature_map(inputs, k_size=self.k_size, strides=self.strides, p_width=pad_w if self.is_add_padding else None)
        if self.next is not None:
            # 切割圖像與卷積核內積
            self.next.forward(np.tensordot(self.split_imgs, self.weights, axes=((3, 4, 5), (1, 2, 3))) + self.bias.T)

    def backward(self, deviation):
        # 輸出對權重的增量
        self.weights_increments = np.tensordot(deviation, self.split_imgs, axes=((0, 1, 2), (0, 1, 2))) / (deviation.shape[0] * self.k_size * self.k_size)
        self.bias_increments = np.sum(np.tensordot(deviation, np.ones((*self.split_imgs.shape[:-1], 1)), axes=((0, 1, 2), (0, 1, 2))), axis=(1, 2)) / (deviation.shape[0] * self.k_size * self.k_size)

        if self.previous is not None:
            # 往前一層的偏差
            pad_w = self.k_size - 1
            pad_w = pad_w // 2 if self.is_add_padding else pad_w
            split_deviation = Conv.split_feature_map(deviation, self.k_size, self.strides, p_width=pad_w)
            self.previous.backward(np.tensordot(split_deviation, np.rot90(self.weights, 2).T, axes=((3, 4, 5), (1, 2, 3))))


