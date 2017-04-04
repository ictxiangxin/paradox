from paradox.kernel.symbol import Symbol, as_symbols


def convolution_1d(data, kernel, mode):
    from paradox.neural_network.convolutional_neural_network.operator import Convolution1D
    return Symbol(operator=Convolution1D(mode), inputs=as_symbols([data, kernel]))


def convolution_2d(data, kernel, mode):
    from paradox.neural_network.convolutional_neural_network.operator import Convolution2D
    return Symbol(operator=Convolution2D(mode), inputs=as_symbols([data, kernel]))
