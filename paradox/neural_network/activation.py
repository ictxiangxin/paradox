from abc import abstractmethod
from paradox.kernel import *
from paradox.utils import xavier_initialization, he_initialization


class ActivationLayer:
    @abstractmethod
    def activation_function(self, *args, **kwargs):
        pass

    @abstractmethod
    def weight_initialization(self, *args, **kwargs):
        pass


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
        output_symbol = exp_symbol / reduce_sum(exp_symbol, axis=0)
        return output_symbol

    @staticmethod
    def weight_initialization(shape):
        weight = xavier_initialization(shape)
        return weight


class HyperbolicTangent(ActivationLayer):
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = tanh(input_symbol)
        return output_symbol

    @staticmethod
    def weight_initialization(shape):
        weight = xavier_initialization(shape)
        return weight


class Sigmoid(ActivationLayer):
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = 1 / (1 + exp(-input_symbol))
        return output_symbol

    @staticmethod
    def weight_initialization(shape):
        weight = xavier_initialization(shape)
        return weight


relu = RectifiedLinearUnits.activation_function
softmax = SoftMax.activation_function
sigmoid = Sigmoid.activation_function


activation_map = {
    'relu': RectifiedLinearUnits,
    'softmax': SoftMax,
    'tanh': HyperbolicTangent,
    'sigmoid': Sigmoid,
}


def register_activation(name: str, activation: ActivationLayer):
    activation_map[name.lower()] = activation


class Activation:
    def __init__(self, name: str):
        self.__name = name.lower()
        self.__activation = None
        if self.__name in activation_map:
            self.__activation = activation_map[self.__name]
        else:
            raise ValueError('No such activation: {}'.format(name))

    def activation_function(self):
        return self.__activation.activation_function

    def weight_initialization(self):
        return self.__activation.weight_initialization
