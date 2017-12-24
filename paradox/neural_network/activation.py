from abc import abstractmethod
from paradox.kernel import *
from paradox.utils import xavier_initialization, he_initialization, normal_initialization


class ActivationLayer:
    def activation_layer(self):
        return self

    @abstractmethod
    def activation_function(self, *args, **kwargs):
        pass

    @staticmethod
    def weight_initialization(shape):
        weight = xavier_initialization(shape)
        return weight

    @staticmethod
    def bias_initialization(shape):
        bias = normal_initialization(shape)
        return bias


class RectifiedLinearUnits(ActivationLayer):
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = maximum(input_symbol, 0)
        return output_symbol

    @staticmethod
    def weight_initialization(shape):
        weight = he_initialization(shape)
        return weight


class SoftMax(ActivationLayer):
    @staticmethod
    def activation_function(input_symbol: Symbol):
        exp_symbol = exp(input_symbol)
        output_symbol = exp_symbol / reduce_sum(exp_symbol, axis=1)
        return output_symbol


class HyperbolicTangent(ActivationLayer):
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = tanh(input_symbol)
        return output_symbol


class Sigmoid(ActivationLayer):
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = 1 / (1 + exp(-input_symbol))
        return output_symbol


class SoftPlus(ActivationLayer):
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = log(1 + exp(input_symbol))
        return output_symbol


class SoftSign(ActivationLayer):
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = input_symbol / (absolute(input_symbol) + 1)
        return output_symbol


relu = RectifiedLinearUnits.activation_function
softmax = SoftMax.activation_function
sigmoid = Sigmoid.activation_function
softplus = SoftPlus.activation_function
softsign = SoftSign.activation_function


activation_map = {
    'relu': RectifiedLinearUnits,
    'softmax': SoftMax,
    'tanh': HyperbolicTangent,
    'sigmoid': Sigmoid,
    'softplus': SoftPlus,
    'softsign': SoftSign,
}


def register_activation(name: str, activation: ActivationLayer):
    activation_map[name.lower()] = activation


class Activation:
    def __init__(self, name: str, *args, **kwargs):
        self.__name = name.lower()
        self.__activation = None
        if self.__name in activation_map:
            self.__activation = activation_map[self.__name](*args, **kwargs)
        else:
            raise ValueError('No such activation: {}'.format(name))

    def activation_layer(self):
        return self.__activation
