import numpy
from paradox.kernel import *


def xavier_initialization(shape):
    weight = numpy.random.randn(*shape) / numpy.sqrt(shape[0])
    return weight


def he_initialization(shape):
    weight = numpy.random.randn(*shape) / numpy.sqrt(shape[0] / 2)
    return weight


class RectifiedLinearUnits:
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = maximum(input_symbol, 0)
        return output_symbol

    @staticmethod
    def weight_initialization(shape):
        weight = he_initialization(shape)
        return weight


class SoftMax:
    @staticmethod
    def activation_function(input_symbol: Symbol):
        exp_symbol = exp(input_symbol)
        output_symbol = exp_symbol / reduce_sum(exp_symbol, axis=0)
        return output_symbol

    @staticmethod
    def weight_initialization(shape):
        weight = xavier_initialization(shape)
        return weight


class HyperbolicTangent:
    @staticmethod
    def activation_function(input_symbol: Symbol):
        output_symbol = tanh(input_symbol)
        return output_symbol

    @staticmethod
    def weight_initialization(shape):
        weight = xavier_initialization(shape)
        return weight


class Sigmoid:
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
