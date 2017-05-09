from paradox.kernel.symbol import Symbol, SymbolCategory, as_symbols


def convolution_nd(data, kernel, dimension: int, mode, element_wise: bool=False):
    from paradox.neural_network.convolutional_neural_network.operator import ConvolutionND
    return Symbol(operator=ConvolutionND(dimension, mode, element_wise), inputs=as_symbols([data, kernel]))


def convolution_1d(data, kernel, mode, element_wise: bool=False):
    return convolution_nd(data, kernel, 1, mode, element_wise)


def convolution_2d(data, kernel, mode, element_wise: bool=False):
    return convolution_nd(data, kernel, 2, mode, element_wise)


def convolution_3d(data, kernel, mode, element_wise: bool=False):
    return convolution_nd(data, kernel, 3, mode, element_wise)


def max_pooling_nd(data, size: tuple, step: tuple, dimension: int, reference=None):
    if reference is None:
        from paradox.neural_network.convolutional_neural_network.operator import MaxPoolingND
        return Symbol(operator=MaxPoolingND(dimension, size, step), inputs=as_symbols([data]))
    else:
        from paradox.neural_network.convolutional_neural_network.operator import MaxReferencePoolingND
        return Symbol(operator=MaxReferencePoolingND(dimension, size, step), inputs=as_symbols([data, reference]))


def max_pooling_1d(data, size: tuple, step: tuple, reference=None):
    return max_pooling_nd(data, size, step, 1, reference)


def max_pooling_2d(data, size: tuple, step: tuple, reference=None):
    return max_pooling_nd(data, size, step, 2, reference)


def max_pooling_3d(data, size: tuple, step: tuple, reference=None):
    return max_pooling_nd(data, size, step, 3, reference)


def max_unpooling_nd(data, pooling, size: tuple, step: tuple, dimension: int):
    from paradox.neural_network.convolutional_neural_network.operator import MaxUnpoolingND
    return Symbol(operator=MaxUnpoolingND(dimension, size, step), inputs=as_symbols([data, pooling]))


def max_unpooling_1d(data, pooling, size: tuple, step: tuple):
    return max_unpooling_nd(data, pooling, size, step, 1)


def max_unpooling_2d(data, pooling, size: tuple, step: tuple):
    return max_unpooling_nd(data, pooling, size, step, 2)


def max_unpooling_3d(data, pooling, size: tuple, step: tuple):
    return max_unpooling_nd(data, pooling, size, step, 3)


def average_pooling_nd(data, size: tuple, step: tuple, dimension: int):
    from paradox.neural_network.convolutional_neural_network.operator import AveragePoolingND
    return Symbol(operator=AveragePoolingND(dimension, size, step), inputs=as_symbols([data]))


def average_pooling_1d(data, size: tuple, step: tuple):
    return average_pooling_nd(data, size, step, 1)


def average_pooling_2d(data, size: tuple, step: tuple):
    return average_pooling_nd(data, size, step, 2)


def average_pooling_3d(data, size: tuple, step: tuple):
    return average_pooling_nd(data, size, step, 3)


def average_unpooling_nd(pooling, size: tuple, step: tuple, dimension: int, unpooling_size: int=None):
    from paradox.neural_network.convolutional_neural_network.operator import AverageUnpoolingND
    return Symbol(operator=AverageUnpoolingND(dimension, size, step, unpooling_size), inputs=as_symbols([pooling]))


def average_unpooling_1d(pooling, size: tuple, step: tuple, unpooling_size: tuple=None):
    return average_unpooling_nd(pooling, size, step, 1, unpooling_size)


def average_unpooling_2d(pooling, size: tuple, step: tuple, unpooling_size: tuple=None):
    return average_unpooling_nd(pooling, size, step, 2, unpooling_size)


def average_unpooling_3d(pooling, size: tuple, step: tuple, unpooling_size: tuple=None):
    return average_unpooling_nd(pooling, size, step, 3, unpooling_size)
