from paradox.kernel.symbol import Symbol, as_symbols


def convolution_1d(data, kernel, mode):
    from paradox.neural_network.convolutional_neural_network.operator import Convolution1D
    return Symbol(operator=Convolution1D(mode), inputs=as_symbols([data, kernel]))


def convolution_2d(data, kernel, mode):
    from paradox.neural_network.convolutional_neural_network.operator import Convolution2D
    return Symbol(operator=Convolution2D(mode), inputs=as_symbols([data, kernel]))


def max_pooling_1d(data, size: int, step: int, reference=None):
    if reference is None:
        from paradox.neural_network.convolutional_neural_network.operator import MaxPooling1D
        return Symbol(operator=MaxPooling1D(size, step), inputs=as_symbols([data]))
    else:
        from paradox.neural_network.convolutional_neural_network.operator import MaxReferencePooling1D
        return Symbol(operator=MaxReferencePooling1D(size, step), inputs=as_symbols([data, reference]))


def max_pooling_2d(data, size: tuple, step: tuple, reference=None):
    if reference is None:
        from paradox.neural_network.convolutional_neural_network.operator import MaxPooling2D
        return Symbol(operator=MaxPooling2D(size, step), inputs=as_symbols([data]))
    else:
        from paradox.neural_network.convolutional_neural_network.operator import MaxReferencePooling2D
        return Symbol(operator=MaxReferencePooling2D(size, step), inputs=as_symbols([data, reference]))


def max_unpooling_1d(data, pooling, size: int, step: int):
    from paradox.neural_network.convolutional_neural_network.operator import MaxUnpooling1D
    return Symbol(operator=MaxUnpooling1D(size, step), inputs=as_symbols([data, pooling]))


def max_unpooling_2d(data, pooling, size: tuple, step: tuple):
    from paradox.neural_network.convolutional_neural_network.operator import MaxUnpooling2D
    return Symbol(operator=MaxUnpooling2D(size, step), inputs=as_symbols([data, pooling]))
