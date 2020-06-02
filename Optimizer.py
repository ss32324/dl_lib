from .Base import np, Base
from .Layer import Normalization, FullyConnection

class Optimizer(Base):
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
        self.rm_list = []
        self.t = 0

    def _calc_new_increments(self, layer, var_name: str):
        return vars(layer)[var_name]

    def clean_layer_val(self, layers):
        for layer in layers:
            for rm_var in self.rm_list:
                if hasattr(layer, rm_var):
                    del vars(layer)[rm_var]
        self.rm_list = []
        self.t = 0

    def gd(self, deviation, layers):
        self.t += 1
        layers[-1].backward(deviation)
        for layer in layers:
            if isinstance(layer, Normalization):
                layer.gamma += self.learning_rate * self._calc_new_increments(layer, 'gamma_increments')
                layer.beta += self.learning_rate * self._calc_new_increments(layer, 'beta_increments')
                # layer.gamma += self.learning_rate * layer.gamma_increments
                # layer.beta += self.learning_rate * layer.beta_increments

            if isinstance(layer, FullyConnection):
                layer.weights += self.learning_rate * self._calc_new_increments(layer, 'weights_increments')
                layer.bias += self.learning_rate * self._calc_new_increments(layer, 'bias_increments')


class SGD(Optimizer):
    ### ref. https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
    def __init__(self, learning_rate=0.0001):
        super().__init__(learning_rate=learning_rate)


class MBGD(SGD):
    ### ref. https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate=learning_rate)

    def _calc_new_increments(self, layer, var_name: str):
        increments = super()._calc_new_increments(layer, var_name)
        return increments / increments.shape[0]


class Adam(MBGD):
    ### ref. https://ruder.io/optimizing-gradient-descent/index.html#adam
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2

    def _calc_new_increments(self, layer, var_name: str):
        increments = super()._calc_new_increments(layer, var_name)
        if not hasattr(layer, var_name + '_m'):
            self.rm_list.extend((var_name + '_m', var_name + '_v'))
            vars(layer)[var_name + '_m'] = vars(layer)[var_name + '_v'] = 0
        vars(layer)[var_name + '_m'] = self.beta1 * vars(layer)[var_name + '_m'] + (1 - self.beta1) * increments
        vars(layer)[var_name + '_v'] = self.beta2 * vars(layer)[var_name + '_v'] + (1 - self.beta2) * (increments** 2)
        m_hat = vars(layer)[var_name + '_m'] / (1 - self.beta1 ** self.t)
        v_hat = vars(layer)[var_name + '_v'] / (1 - self.beta2 ** self.t)
        return m_hat / ((v_hat ** 0.5) + Adam.CONST_E)

