from .Base import Base, np

class Layer(Base):
    is_train = False
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

    def act(self, x):
        return np.nan_to_num(x)

    def dact(self, y):
        return np.nan_to_num(y)

    def forward(self, x):
        self.outputs = self.act(x)
        if self.next is not None:
            self.next.forward(self.outputs)

    def backward(self, deviation):
        if self.previous is not None:
            if not self.is_pair_with_lossfunc:
                bk_deviation = deviation * self.dact(self.outputs)
                self.previous.backward(bk_deviation)
            else:
                self.previous.backward(deviation)

### Normalization Func
class Normalization(Layer):
    def __init__(self, mov=0.9):
        super().__init__()
        self.gamma = 1
        self.beta = 0
        self.mov = mov
        self.mean = self.mov_mean = 0
        self.var = self.mov_var = 0

    def setAxis(self, inputs):
        return [-1]

    def forward(self, inputs):
        self.axis = self.setAxis(inputs)
        self.inputs = inputs
        ### 批量均值
        self.mean = np.mean(inputs, axis=self.axis, keepdims=True)
        ### 批量方差
        self.var = np.mean((inputs - self.mean)** 2, axis=self.axis, keepdims=True)
        ### 數值依均值位移後 依方差為準值縮小
        if Layer.is_train:
            self.z_hat = (inputs - self.mean) * (self.var + Layer.CONST_E)** -0.5
        else:
            try:
                self.z_hat = (inputs - self.mov_mean) * (self.mov_var + Layer.CONST_E)** -0.5
            except:
                self.z_hat = (inputs - self.mean) * (self.var + Layer.CONST_E)** -0.5
        if self.next is not None:
            self.next.forward(self.z_hat)

    def backward(self, deviation):
        ### 更新
        if Layer.is_train:
            self.mov_mean = self.mov * self.mov_mean + (1 - self.mov) * self.mean
            self.mov_var = self.mov * self.mov_var + (1 - self.mov) * self.var

        if self.previous is not None:
            ### ref. https://towardsdatascience.com/implementing-spatial-batch-instance-layer-normalization-in-tensorflow-manual-back-prop-in-tf-77faa8d2c362
            cg_prts = 1.0 / np.prod(np.array(deviation.shape[0:len(self.axis)]))
            gsigma = np.sum(deviation * (self.inputs - self.mean), axis=self.axis, keepdims=True) * -0.5 * (self.var + Layer.CONST_E)** -1.5
            gmu = np.sum(deviation * -(self.var + Layer.CONST_E)** -0.5, axis=self.axis, keepdims=True) - gsigma * cg_prts * 2.0 * np.sum((self.inputs - self.mean), axis=self.axis, keepdims=True)
            bk_deviation = deviation * (self.var + Layer.CONST_E)** -0.5 + gsigma * cg_prts * 2.0 * np.sum((self.inputs - self.mean), axis=self.axis, keepdims=True) + gmu * cg_prts
            self.previous.backward(bk_deviation)

### Pool Layer
class Pool(Layer):

    def __init__(self, p_size):
        super().__init__()
        self.p_size = p_size

    def cg_img_hw(self, x):
        ### 若圖片長寬為奇數
        img_height, img_width = x.shape[1:3]
        if img_height % self.p_size != 0:
            x = x[:,:img_height - (img_height % self.p_size),:,:]
        if img_width % self.p_size != self.p_size:
            x = x[:,:,:img_width - (img_width % self.p_size),:]
        self.cg_shape = x.shape
        return x

    def bcg_img_hw(self, x):
        img_height, img_width = x.shape[1:3]
        ### 還原補0
        if self.shape[1] != x.shape[1]:
            x = np.pad(x, mode='constant', pad_width=((0, 0), (0, (img_height % self.p_size)), (0, 0), (0, 0)))
        if self.shape[2] != x.shape[2]:
            x = np.pad(x, mode='constant', pad_width=((0, 0), (0, 0), (0, (img_width % self.p_size)), (0, 0)))
        return x

    def rule(self, sp_img):
        pool = np.max(sp_img, axis=(2, 4), keepdims=True)
        pool_shape = pool.shape
        mask = np.where(pool == sp_img, 1, 0)
        pool = pool.reshape((pool.shape[0], pool.shape[1], pool.shape[3], sp_img.shape[-1]))
        return pool, pool_shape, mask

    def forward(self, inputs):
        self.shape = self.cg_shape = inputs.shape
        inputs = self.cg_img_hw(inputs)
        img_height, img_width, img_ch = inputs.shape[1:]
        sp_height, sp_width = img_height // self.p_size, img_width // self.p_size
        # 切割
        split_imgs = inputs.reshape((inputs.shape[0], sp_height, self.p_size, sp_width, self.p_size, img_ch))
        pool, self.pool_shape, self.mask = self.rule(split_imgs)
        if self.next is not None:
            self.next.forward(pool)

    def backward(self, deviation):
        if self.previous is not None:
            bk_deviation = deviation.reshape(self.pool_shape) * self.mask
            bk_deviation = self.bcg_img_hw(bk_deviation.reshape(self.cg_shape))
            self.previous.backward(bk_deviation)

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
        if Layer.is_train:
            self.mask = (np.random.rand(*inputs.shape) > self.rate)
            o = self.mask * inputs
        else:
            o = inputs
        self.next.forward(o)

    def backward(self, deviation):
        if self.previous is not None:
            if Layer.is_train:
                self.previous.backward(deviation * self.mask)
            else:
                self.previous.backward(deviation)
