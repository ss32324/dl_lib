from ..Base import np
from ..Layer import Pool

class MaxPooling(Pool):
    ### ref. https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
    def rule(self, sp_img):
        pool = np.max(sp_img, axis=(2, 4), keepdims=True)
        pool_shape = pool.shape
        mask = np.where(pool == sp_img, 1, 0)
        pool = pool.reshape((pool.shape[0], pool.shape[1], pool.shape[3], sp_img.shape[-1]))
        return pool, pool_shape, mask
