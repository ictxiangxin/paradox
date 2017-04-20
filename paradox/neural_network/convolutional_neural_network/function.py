from paradox.kernel.symbol import Symbol, SymbolCategory, as_symbols


def convolution_nd(data, kernel, dimension: int, mode, element_wise: bool=False):
    from paradox.neural_network.convolutional_neural_network.operator import ConvolutionND
    return Symbol(operator=ConvolutionND(dimension, mode, element_wise), inputs=as_symbols([data, kernel]), category=SymbolCategory.operator)


def convolution_1d(data, kernel, mode, element_wise: bool=False):
    return convolution_nd(data, kernel, 1, mode, element_wise)


def convolution_2d(data, kernel, mode, element_wise: bool=False):
    return convolution_nd(data, kernel, 2, mode, element_wise)


def max_pooling_1d(data, size: int, step: int, reference=None):
    if reference is None:
        from paradox.neural_network.convolutional_neural_network.operator import MaxPooling1D
        return Symbol(operator=MaxPooling1D(size, step), inputs=as_symbols([data]), category=SymbolCategory.operator)
    else:
        from paradox.neural_network.convolutional_neural_network.operator import MaxReferencePooling1D
        return Symbol(operator=MaxReferencePooling1D(size, step), inputs=as_symbols([data, reference]), category=SymbolCategory.operator)


def max_pooling_2d(data, size: tuple, step: tuple, reference=None):
    if reference is None:
        from paradox.neural_network.convolutional_neural_network.operator import MaxPooling2D
        return Symbol(operator=MaxPooling2D(size, step), inputs=as_symbols([data]), category=SymbolCategory.operator)
    else:
        from paradox.neural_network.convolutional_neural_network.operator import MaxReferencePooling2D
        return Symbol(operator=MaxReferencePooling2D(size, step), inputs=as_symbols([data, reference]), category=SymbolCategory.operator)


def max_unpooling_1d(data, pooling, size: int, step: int):
    from paradox.neural_network.convolutional_neural_network.operator import MaxUnpooling1D
    return Symbol(operator=MaxUnpooling1D(size, step), inputs=as_symbols([data, pooling]), category=SymbolCategory.operator)


def max_unpooling_2d(data, pooling, size: tuple, step: tuple):
    from paradox.neural_network.convolutional_neural_network.operator import MaxUnpooling2D
    return Symbol(operator=MaxUnpooling2D(size, step), inputs=as_symbols([data, pooling]), category=SymbolCategory.operator)


def average_pooling_1d(data, size: int, step: int):
    from paradox.neural_network.convolutional_neural_network.operator import AveragePooling1D
    return Symbol(operator=AveragePooling1D(size, step), inputs=as_symbols([data]), category=SymbolCategory.operator)


def average_pooling_2d(data, size: tuple, step: tuple):
    from paradox.neural_network.convolutional_neural_network.operator import AveragePooling2D
    return Symbol(operator=AveragePooling2D(size, step), inputs=as_symbols([data]), category=SymbolCategory.operator)


def average_unpooling_1d(pooling, size: int, step: int, unpooling_size: int=None):
    from paradox.neural_network.convolutional_neural_network.operator import AverageUnpooling1D
    return Symbol(operator=AverageUnpooling1D(size, step, unpooling_size), inputs=as_symbols([pooling]), category=SymbolCategory.operator)


def average_unpooling_2d(pooling, size: tuple, step: tuple, unpooling_size: tuple=None):
    from paradox.neural_network.convolutional_neural_network.operator import AverageUnpooling2D
    return Symbol(operator=AverageUnpooling2D(size, step, unpooling_size), inputs=as_symbols([pooling]), category=SymbolCategory.operator)
