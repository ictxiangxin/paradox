from paradox.kernel import *


def relu(input_symbol: Symbol):
    output_symbol = maximum(input_symbol, 0)
    return output_symbol
