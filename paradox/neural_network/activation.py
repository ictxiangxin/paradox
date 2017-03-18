from paradox.kernel import *


def relu(input_symbol: Symbol):
    output_symbol = maximum(input_symbol, 0)
    return output_symbol


def softmax(input_symbol: Symbol):
    exp_symbol = exp(input_symbol)
    output_symbol = exp_symbol / reduce_sum(exp_symbol, axis=0)
    return output_symbol


def tanh(input_symbol: Symbol):
    a = exp(input_symbol)
    b = exp(-input_symbol)
    output_symbol = (a - b) / (a + b)
    return output_symbol


def sigmoid(input_symbol: Symbol):
    output_symbol = 1 / (1 + exp(-input_symbol))
    return output_symbol


activation_map = {
    'relu': relu,
    'softmax': softmax,
    'tanh': tanh,
    'sigmoid': sigmoid,
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
        return self.__activation
