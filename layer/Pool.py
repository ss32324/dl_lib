from ..Base import np
from ..Layer import Pool

class MaxPooling(Pool):
    ### ref. https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
    def __init__(self, p_size):
        super().__init__()
        self.p_size = p_size

    def cg_img_hw(self, x):
        ### 若圖片長寬為奇數
        img_height, img_width = x.shape[1:3]
        if img_height % 2 == 1:
            x = x[:,:img_height-1,:,:]
        if img_width % 2 == 1:
            x = x[:,:,:img_width - 1,:]
        self.cg_shape = x.shape
        return x

    def bcg_img_hw(self, x):
        ### 還原補0
        if self.shape[1] != x.shape[1]:
            x = np.pad(x, mode='constant', pad_width=((0, 0), (0, 1), (0, 0), (0, 0)))
        if self.shape[2] != x.shape[2]:
            x = np.pad(x, mode='constant', pad_width=((0, 0), (0, 0), (0, 1), (0, 0)))
        return x

    def forward(self, inputs):
        self.shape = self.cg_shape = inputs.shape
        inputs = self.cg_img_hw(inputs)
        img_height, img_width, img_ch = inputs.shape[1:]
        sp_height, sp_width = img_height // self.p_size, img_width // self.p_size
        # 切割
        split_imgs = inputs.reshape((inputs.shape[0], sp_height, self.p_size, sp_width, self.p_size, img_ch))
        # 取切割範圍最大值
        max_pool = np.max(split_imgs, axis=(2, 4), keepdims=True)
        self.max_pool_shape = max_pool.shape
        self.mask = np.where(max_pool == split_imgs, 0, 1)
        max_pool = max_pool.reshape((max_pool.shape[0], max_pool.shape[1], max_pool.shape[3], img_ch))
        if self.next is not None:
            self.next.forward(max_pool)

    def backward(self, deviation):
        if self.previous is not None:
            bk_deviation = deviation.reshape(self.max_pool_shape) * self.mask
            bk_deviation = self.bcg_img_hw(bk_deviation.reshape(self.cg_shape))
            self.previous.backward(bk_deviation)