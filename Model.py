import joblib
from .Base import np
from .Layer import Layer
from .Optimizer import Optimizer
from .LossFunc import LossFunc, CategoricalCrossEntropy, BinaryCrossEntropy
from .layer import Softmax, Sigmoid


class Model:
    def __init__(self, layers: list):
        self.layers = layers
        self.batch_size = 32
        Model._link(self.layers)

    def _link(layers):
        previous = None
        for layer in layers:
            if previous is not None:
                layer.previous = previous
                previous.next = layer
            previous = layer


    def fit(self, train_x, train_y, test_x, test_y, epoch=1, batch_size=32, clean_opt=0):
        self.batch_size = batch_size
        results = {
            'val_acc': [], 'val_los': [],
            'acc': [], 'los': [],
        }
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)

        indexes = np.arange(0, train_x.shape[0])
        # 補齊未滿一個batch_size的量
        indexes = np.hstack((indexes, indexes[:train_x.shape[0] % batch_size]))
        for _i in range(epoch):

            # 隨機索引(打亂資料)
            indexes = np.random.permutation(indexes)

            print('\n<{:02d}> {}'.format(_i+1, '-'*80))
            # training
            accuracy, loss = self._train(train_x[indexes], train_y[indexes])
            results['acc'].append(accuracy)
            results['los'].append(loss)

            # verifying
            accuracy, loss = self.evaluate(test_x, test_y)
            results['val_acc'].append(accuracy)
            results['val_los'].append(loss)

            if clean_opt > 0:
                if epoch % clean_opt == 0:
                    self.optimizer.clean_layer_val(self.layers)

        return results

    def evaluate(self, x, y, batch_size: int=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        x = np.asarray(x)
        y = np.asarray(y)
        iterations = x.shape[0] // batch_size

        accuracy, loss = 0, 0
        for i in range(iterations):
            batch_x = x[i * batch_size:] if i == iterations-1 else x[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:] if i == iterations-1 else y[i * batch_size:(i + 1) * batch_size]
            y_hat = self._forward(batch_x)
            acc, los = self.calc_accuracy(batch_y, y_hat), self.calc_loss(batch_y, y_hat)
            accuracy += float(acc)
            loss += float(los)
            print('    evaluate[{:25}] acc:{:.2f}, loss:{:.3e}'.format('='*(25*i//iterations)+'>', accuracy / (i + 1) * 100, loss / (i + 1)), end='\r')
        print()
        return accuracy / iterations, loss / iterations

    def get_diff(self, x, y, batch_size: int=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        x = np.asarray(x)
        y = np.asarray(y)
        y = y.reshape((*y.shape[0:2])) if y.ndim == 3 else y

        iterations = x.shape[0] // batch_size
        diff_indexes = []
        for i in range(iterations):
            batch_x = x[i * batch_size:] if i == iterations-1 else x[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:] if i == iterations-1 else y[i * batch_size:(i + 1) * batch_size]
            y_hat = self._forward(batch_x)
            diff_indexes.extend((i * batch_size + np.where(np.argmax(batch_y, axis=1) != np.argmax(y_hat, axis=1))[0]).tolist())
            print('    get_diff[{:25}] {}'.format('='*(25*i//iterations)+'>', len(diff_indexes)), end='\r')
        print()
        return diff_indexes

    def predict(self, x):
        x = np.asarray(x)
        return self._forward(x)

    def _forward(self, x):
        self.layers[0].forward(x)
        return self.layers[-1].outputs

    def calc_loss(self, y, y_hat):
        y = y.reshape((*y.shape[0:2])) if y.ndim == 3 else y
        return np.nan_to_num(np.mean(self.lossfunc.calc_loss(y, y_hat)))

    def calc_accuracy(self, y, y_hat):
        y = y.reshape((*y.shape[0:2])) if y.ndim == 3 else y
        compare = np.equal(np.argmax(y_hat, axis=1), np.argmax(y, axis=1))
        return np.nan_to_num(np.mean(compare))

    def compile(self, optimizer: Optimizer, lossfunc: LossFunc):
        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.check_loss_activation_pair()

    def check_loss_activation_pair(self):
        if isinstance(self.lossfunc, CategoricalCrossEntropy) and isinstance(self.layers[-1], Softmax) \
            or isinstance(self.lossfunc, BinaryCrossEntropy) and isinstance(self.layers[-1], Sigmoid):
            self.layers[-1].is_pair_with_lossfunc = True


    def _train(self, train_x, train_y):
        batch_size = self.batch_size
        # 計算迭代次數
        iterations = train_x.shape[0] // batch_size

        accuracy, loss = 0, 0
        for i in range(iterations):
            batch_x = train_x[i * batch_size:(i + 1) * batch_size]
            batch_y = train_y[i * batch_size:(i + 1) * batch_size]

            ### training
            Layer.is_train = True
            # --> 輸入到輸出 -->
            outputs = self._forward(batch_x)
            # <-- 反向傳遞 <--
            deviation = self.lossfunc.backprop(batch_y, outputs)
            self.optimizer.gd(deviation, self.layers)

            ### testing
            Layer.is_train = False
            # --> 輸入到輸出 -->
            outputs = self._forward(batch_x)
            # <-- 反向傳遞 <--
            deviation = self.lossfunc.backprop(batch_y, outputs)
            self.optimizer.gd(deviation, self.layers)

            ### 印出
            acc, los = self.calc_accuracy(batch_y, outputs), self.calc_loss(batch_y, outputs)
            accuracy += float(acc)
            loss += float(los)
            print('    training[{:25}] acc:{:.2f}, loss:{:.3e}'.format('='*(25*i//iterations)+'>', accuracy / (i + 1) * 100, loss / (i + 1)), end='\r')

        print()
        return accuracy / iterations, loss / iterations

    def save(model, path):
        import joblib
        joblib.dump(model, path)

    def load(path):
        import joblib
        return joblib.load(path)
