import numpy
from paradox.kernel import flip, rotate90, reduce_mean, expand
from paradox.kernel.operator import Operator, element_wise_shape
from paradox.neural_network.convolutional_neural_network.compute import \
    ConvolutionMode, \
    compute_convolution_nd, \
    compute_max_pooling_nd, \
    compute_max_unpooling_nd, \
    compute_average_pooling_1d, \
    compute_average_pooling_2d, \
    compute_average_unpooling_1d, \
    compute_average_unpooling_2d
from paradox.neural_network.convolutional_neural_network.function import \
    convolution_nd, \
    max_pooling_nd, \
    max_unpooling_nd, \
    average_pooling_1d, \
    average_pooling_2d, \
    average_unpooling_1d, \
    average_unpooling_2d


def convolution_nd_shape(shape_data, shape_kernel, dimension, mode):
    prefix_shape = shape_data[:-dimension] + shape_kernel[:-dimension]
    if mode == 'valid' or mode == ConvolutionMode.valid:
        new_shape = prefix_shape + tuple(shape_data[i] - shape_kernel[i] + 1 for i in range(-dimension, 0))
    elif mode == 'full' or mode == ConvolutionMode.full:
        new_shape = prefix_shape + tuple(shape_data[i] + shape_kernel[i] - 1 for i in range(-dimension, 0))
    else:
        raise ValueError('Invalid convolution mode: {}'.format(mode))
    return new_shape, (), ()


def element_wise_convolution_nd_shape(shape_data, shape_kernel, dimension, mode):
    prefix_shape, prefix_broadcast_data, prefix_broadcast_kernel = element_wise_shape(shape_data[:-dimension], shape_kernel[:-dimension])
    if mode == 'valid' or mode == ConvolutionMode.valid:
        new_shape = prefix_shape + tuple(shape_data[i] - shape_kernel[i] + 1 for i in range(-dimension, 0))
    elif mode == 'full' or mode == ConvolutionMode.full:
        new_shape = prefix_shape + tuple(shape_data[i] + shape_kernel[i] - 1 for i in range(-dimension, 0))
    else:
        raise ValueError('Invalid convolution mode: {}'.format(mode))
    return new_shape, prefix_broadcast_data + (0,) * dimension, prefix_broadcast_kernel + (0,) * dimension


def pooling_nd_shape(shape_data, size, step, dimension):
    prefix_shape = shape_data[:-dimension]
    if not isinstance(size, tuple):
        size = (size,)
    if not isinstance(step, tuple):
        step = (step,)
    new_shape = prefix_shape + tuple(len(range(0, shape_data[i] - size[i] + 1, step[i])) for i in range(-dimension, 0))
    return new_shape, ()


def max_unpooling_shape(shape_data, shape_pooling, dimension):
    prefix_shape, prefix_broadcast_data, prefix_broadcast_kernel = element_wise_shape(shape_data[:-dimension], shape_pooling[:-dimension])
    new_shape = prefix_shape + tuple(shape_data[i] for i in range(-dimension, 0))
    return new_shape, prefix_broadcast_data + (0,) * dimension, prefix_broadcast_kernel + (0,) * dimension


def unpooling_shape(shape_pooling, size, step, unpooling_size, dimension):
    prefix_shape, prefix_broadcast_data, prefix_broadcast_kernel = element_wise_shape(shape_pooling[:-dimension], shape_pooling[:-dimension])
    new_shape = prefix_shape + tuple((size[i] + (shape_pooling[i] - 1) * step[i]) if unpooling_size is None else unpooling_size[i] for i in range(-dimension, 0))
    return new_shape, prefix_broadcast_data + (0,) * dimension, prefix_broadcast_kernel + (0,) * dimension


class ConvolutionND(Operator):
    def __init__(self, dimension: int, mode, element_wise: bool=False):
        self.inputs_count = 2
        self.arguments = {'dimension': dimension, 'mode': mode, 'element_wise': element_wise}
        assert dimension > 0

    def compute(self, value_data, value_kernel):
        return compute_convolution_nd(value_data, value_kernel, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_kernel):
        forward = engine.gradient(symbol_forward)
        dimension = self.arguments['dimension']
        mode = self.arguments['mode']
        if mode == 'valid' or mode == ConvolutionMode.valid:
            prefix_shape_kernel = engine.shape(symbol_kernel)[:-dimension]
            prefix_shape_data = engine.shape(symbol_data)[:-dimension]
            if dimension == 1:
                flip_kernel = flip(symbol_kernel, -1)
            else:
                flip_kernel = rotate90(symbol_kernel, count=2, axes=(-2, -1))
            gradient_data = convolution_nd(forward, flip_kernel, dimension, ConvolutionMode.full, True)
            for _ in prefix_shape_kernel:
                gradient_data = reduce_mean(gradient_data, axis=-dimension - 1)
                symbol_data = expand(symbol_data, -dimension - 1)
            gradient_kernel = convolution_nd(symbol_data, forward, dimension, ConvolutionMode.valid, True)
            for _ in prefix_shape_data:
                gradient_kernel = reduce_mean(gradient_kernel, axis=-dimension - 1 - len(prefix_shape_kernel))
            return [lambda: gradient_data,
                    lambda: gradient_kernel]
        elif mode == 'full' or mode == ConvolutionMode.full:
            prefix_shape_kernel = engine.shape(symbol_kernel)[:-dimension]
            prefix_shape_data = engine.shape(symbol_data)[:-dimension]
            if dimension == 1:
                flip_kernel = flip(symbol_kernel, -1)
            else:
                flip_kernel = rotate90(symbol_kernel, count=2, axes=(-2, -1))
            gradient_data = convolution_nd(forward, flip_kernel, dimension, ConvolutionMode.valid, True)
            for _ in prefix_shape_kernel:
                gradient_data = reduce_mean(gradient_data, axis=-dimension - 1)
                symbol_data = expand(symbol_data, -dimension - 1)
            flip_gradient = convolution_nd(forward, symbol_data, dimension, ConvolutionMode.valid, True)
            if dimension == 1:
                gradient_kernel = flip(flip_gradient, -1)
            else:
                gradient_kernel = rotate90(flip_gradient, count=2, axes=(-2, -1))
            for _ in prefix_shape_data:
                gradient_kernel = reduce_mean(gradient_kernel, axis=-dimension - 1 - len(prefix_shape_kernel))
            return [lambda: gradient_data,
                    lambda: gradient_kernel]
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    def shape(self, shape_data, shape_kernel):
        if self.arguments['element_wise']:
            return element_wise_convolution_nd_shape(shape_data, shape_kernel, self.arguments['dimension'], self.arguments['mode'])
        else:
            return convolution_nd_shape(shape_data, shape_kernel, self.arguments['dimension'], self.arguments['mode'])


class Convolution1D(ConvolutionND):
    def __init__(self, mode, element_wise: bool=False):
        ConvolutionND.__init__(self, 1, mode, element_wise)


class Convolution2D(ConvolutionND):
    def __init__(self, mode, element_wise: bool=False):
        ConvolutionND.__init__(self, 2, mode, element_wise)


class MaxPoolingND(Operator):
    def __init__(self, dimension: int, size: tuple, step: tuple):
        self.inputs_count = 1
        self.arguments = {'dimension': dimension, 'size': size, 'step': step}

    def compute(self, value_data):
        return compute_max_pooling_nd(value_data, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_nd(symbol_data, forward, **self.arguments)]

    def shape(self, shape_data):
        return pooling_nd_shape(shape_data, **self.arguments)


class MaxPooling1D(MaxPoolingND):
    def __init__(self, size: tuple, step: tuple):
        MaxPoolingND.__init__(self, 1, size, step)


class MaxPooling2D(MaxPoolingND):
    def __init__(self, size: tuple, step: tuple):
        MaxPoolingND.__init__(self, 2, size, step)


class MaxReferencePoolingND(Operator):
    def __init__(self, dimension: int, size: tuple, step: tuple):
        self.inputs_count = 2
        self.arguments = {'dimension': dimension, 'size': size, 'step': step}

    def compute(self, value_data, reference_data):
        return compute_max_pooling_nd(value_data, reference=reference_data, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_reference):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_nd(symbol_reference, forward, **self.arguments),
                lambda: max_unpooling_nd(symbol_reference, numpy.ones(engine.shape(forward)), **self.arguments)]

    def shape(self, shape_data):
        return pooling_nd_shape(shape_data, **self.arguments)


class MaxReferencePooling1D(MaxReferencePoolingND):
    def __init__(self, size: tuple, step: tuple):
        MaxReferencePoolingND.__init__(self, 1, size, step)


class MaxReferencePooling2D(MaxReferencePoolingND):
    def __init__(self, size: tuple, step: tuple):
        MaxReferencePoolingND.__init__(self, 2, size, step)


class MaxUnpoolingND(Operator):
    def __init__(self, dimension: int, size: tuple, step: tuple):
        self.inputs_count = 2
        self.arguments = {'dimension': dimension, 'size': size, 'step': step}

    def compute(self, value_data, value_pooling):
        return compute_max_unpooling_nd(value_data, value_pooling, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_pooling):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_nd(symbol_data, numpy.ones(engine.shape(symbol_pooling)), **self.arguments),
                lambda: max_pooling_nd(forward, reference=symbol_data, **self.arguments)]

    def shape(self, shape_data, shape_pooling):
        return max_unpooling_shape(shape_data, shape_pooling, self.arguments['dimension'])


class MaxUnpooling1D(MaxUnpoolingND):
    def __init__(self, size: tuple, step: tuple):
        MaxUnpoolingND.__init__(self, 1, size, step)


class MaxUnpooling2D(MaxUnpoolingND):
    def __init__(self, size: tuple, step: tuple):
        MaxUnpoolingND.__init__(self, 2, size, step)


class AveragePooling1D(Operator):
    def __init__(self, size: int, step: int):
        self.inputs_count = 1
        self.arguments = {'size': size, 'step': step}

    def compute(self, value_data):
        return compute_average_pooling_1d(value_data, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data):
        forward = engine.gradient(symbol_forward)
        return [lambda: average_unpooling_1d(forward, unpooling_size=engine.shape(symbol_data)[-1], **self.arguments)]

    def shape(self, shape_data):
        return pooling_nd_shape(shape_data, dimension=1, **self.arguments)


class AveragePooling2D(Operator):
    def __init__(self, size: tuple, step: tuple):
        self.inputs_count = 1
        self.arguments = {'size': size, 'step': step}

    def compute(self, value_data):
        return compute_average_pooling_2d(value_data, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data):
        forward = engine.gradient(symbol_forward)
        return [lambda: average_unpooling_2d(forward, unpooling_size=engine.shape(symbol_data)[-2:], **self.arguments)]

    def shape(self, shape_data):
        return pooling_nd_shape(shape_data, dimension=2, **self.arguments)


class AverageUnpooling1D(Operator):
    def __init__(self, size: int, step: int, unpooling_size: int=None):
        self.inputs_count = 1
        self.arguments = {'size': size, 'step': step, 'unpooling_size': unpooling_size}

    def compute(self, value_pooling):
        return compute_average_unpooling_1d(value_pooling, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_pooling):
        forward = engine.gradient(symbol_forward)
        return [lambda: average_pooling_1d(forward, **self.arguments)]

    def shape(self, shape_data):
        return unpooling_shape(shape_data, dimension=1, **self.arguments)


class AverageUnpooling2D(Operator):
    def __init__(self, size: tuple, step: tuple, unpooling_size: tuple=None):
        self.inputs_count = 1
        self.arguments = {'size': size, 'step': step, 'unpooling_size': unpooling_size}

    def compute(self, value_pooling):
        return compute_average_unpooling_2d(value_pooling, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_pooling):
        forward = engine.gradient(symbol_forward)
        return [lambda: average_pooling_2d(forward, **self.arguments)]

    def shape(self, shape_data):
        return unpooling_shape(shape_data, dimension=2, **self.arguments)
