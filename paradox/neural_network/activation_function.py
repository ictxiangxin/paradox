from paradox.kernel import *


def relu(input_symbol: Symbol):
    output_symbol = maximum(input_symbol, 0)
    return output_symbol


def softmax(input_symbol: Symbol):
    exp_symbol = exp(input_symbol)
    output_symbol = exp_symbol / reduce_sum(exp_symbol, axis=0)
    return output_symbol
