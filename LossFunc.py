from .Base import np, Base

### Loss Func
class LossFunc(Base):
    pass

class MSE(LossFunc):

    def backprop(self, y, y_hat):
        y = y.reshape((*y.shape[0:2])) if y.ndim == 3 else y
        return 2 * (y - y_hat)

    def calc_loss(self, y, y_hat):
        loss = np.mean((y - y_hat)** 2, axis=1)
        return np.nan_to_num(loss)


class CategoricalCrossEntropy(LossFunc):

    def backprop(self, y, y_hat):
        y = y.reshape((*y.shape[0:2])) if y.ndim == 3 else y
        return y - y_hat

    def calc_loss(self, y, y_hat):
        log_y_hat = np.nan_to_num(np.log(y_hat))
        loss = -np.sum(y * log_y_hat, axis=1)
        return np.nan_to_num(loss)


class BinaryCrossEntropy(LossFunc):

    def backprop(self, y, y_hat):
        y = y.reshape((*y.shape[0:2])) if y.ndim == 3 else y
        return y - y_hat

    def calc_loss(self, y, y_hat):
        log_y_hat = np.nan_to_num(np.log(y_hat))
        log_1m_y_hat = np.nan_to_num(np.log(1 - y_hat))
        loss = -np.sum(y * log_y_hat + (1 - y) * log_1m_y_hat, axis=1)
        return np.nan_to_num(loss)
