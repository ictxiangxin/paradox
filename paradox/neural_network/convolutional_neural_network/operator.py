import numpy
from paradox.kernel import flip, rotate90
from paradox.kernel.operator import Operator, element_wise_shape
from paradox.neural_network.convolutional_neural_network.compute import \
    ConvolutionMode, \
    compute_convolution_1d, \
    compute_convolution_2d, \
    compute_max_pooling_1d, \
    compute_max_pooling_2d, \
    compute_max_unpooling_1d, \
    compute_max_unpooling_2d, \
    compute_average_pooling_1d, \
    compute_average_unpooling_1d
from paradox.neural_network.convolutional_neural_network.function import \
    convolution_1d, \
    convolution_2d, \
    max_pooling_1d, \
    max_pooling_2d, \
    max_unpooling_1d, \
    max_unpooling_2d, \
    average_pooling_1d, \
    average_unpooling_1d


def convolution_shape(shape_data, shape_kernel, mode, dimension):
    prefix_shape, prefix_broadcast_data, prefix_broadcast_kernel = element_wise_shape(shape_data[:-dimension], shape_kernel[:-dimension])
    if mode == 'valid' or mode == ConvolutionMode.valid:
        new_shape = prefix_shape + tuple(shape_data[i] - shape_kernel[i] + 1 for i in range(-dimension, 0))
    elif mode == 'full' or mode == ConvolutionMode.full:
        new_shape = prefix_shape + tuple(shape_data[i] + shape_kernel[i] - 1 for i in range(-dimension, 0))
    else:
        raise ValueError('Invalid convolution mode: {}'.format(mode))
    return new_shape, prefix_broadcast_data + (0,) * dimension, prefix_broadcast_kernel + (0,) * dimension


def pooling_shape(shape_data, size, step, dimension):
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


def average_unpooling_shape(shape_pooling, size, step, unpooling_size, dimension):
    prefix_shape, prefix_broadcast_data, prefix_broadcast_kernel = element_wise_shape(shape_pooling[:-dimension], shape_pooling[:-dimension])
    new_shape = prefix_shape + tuple((size + (shape_pooling[i] - 1) * step) if unpooling_size is None else unpooling_size[i] for i in range(-dimension, 0))
    return new_shape, prefix_broadcast_data + (0,) * dimension, prefix_broadcast_kernel + (0,) * dimension


class Convolution1D(Operator):
    def __init__(self, mode):
        self.inputs_count = 2
        self.arguments = {'mode': mode}

    def compute(self, value_data, value_kernel):
        return compute_convolution_1d(value_data, value_kernel, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_kernel):
        forward = engine.gradient(symbol_forward)
        mode = self.arguments['mode']
        if mode == 'valid' or mode == ConvolutionMode.valid:
            return [lambda: convolution_1d(forward, flip(symbol_kernel, -1), ConvolutionMode.full),
                    lambda: convolution_1d(symbol_data, forward, ConvolutionMode.valid)]
        elif mode == 'full' or mode == ConvolutionMode.full:
            return [lambda: convolution_1d(forward, flip(symbol_kernel, -1), ConvolutionMode.valid),
                    lambda: flip(convolution_1d(forward, symbol_data, ConvolutionMode.valid), -1)]
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    def shape(self, shape_data, shape_kernel):
        return convolution_shape(shape_data, shape_kernel, self.arguments['mode'], 1)


class Convolution2D(Operator):
    def __init__(self, mode):
        self.inputs_count = 2
        self.arguments = {'mode': mode}

    def compute(self, value_data, value_kernel):
        return compute_convolution_2d(value_data, value_kernel, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_kernel):
        forward = engine.gradient(symbol_forward)
        mode = self.arguments['mode']
        if mode == 'valid' or mode == ConvolutionMode.valid:
            return [lambda: convolution_2d(forward, rotate90(symbol_kernel, count=2, axes=(-2, -1)), ConvolutionMode.full),
                    lambda: convolution_2d(symbol_data, forward, ConvolutionMode.valid)]
        elif mode == 'full' or mode == ConvolutionMode.full:
            return [lambda: convolution_2d(forward, rotate90(symbol_kernel, count=2, axes=(-2, -1)), ConvolutionMode.valid),
                    lambda: rotate90(convolution_2d(forward, symbol_data, ConvolutionMode.valid), count=2, axes=(-2, -1))]
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

    def shape(self, shape_data, shape_kernel):
        return convolution_shape(shape_data, shape_kernel, self.arguments['mode'], 2)


class MaxPooling1D(Operator):
    def __init__(self, size: int, step: int):
        self.inputs_count = 1
        self.arguments = {'size': size, 'step': step}

    def compute(self, value_data):
        return compute_max_pooling_1d(value_data, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_1d(symbol_data, forward, **self.arguments)]

    def shape(self, shape_data):
        return pooling_shape(shape_data, dimension=1, **self.arguments)


class MaxPooling2D(Operator):
    def __init__(self, size: tuple, step: tuple):
        self.inputs_count = 1
        self.arguments = {'size': size, 'step': step}

    def compute(self, value_data):
        return compute_max_pooling_2d(value_data, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_2d(symbol_data, forward, **self.arguments)]

    def shape(self, shape_data):
        return pooling_shape(shape_data, dimension=2, **self.arguments)


class MaxReferencePooling1D(Operator):
    def __init__(self, size: int, step: int):
        self.inputs_count = 2
        self.arguments = {'size': size, 'step': step}

    def compute(self, value_data, reference_data):
        return compute_max_pooling_1d(value_data, reference=reference_data, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_reference):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_1d(symbol_reference, forward, **self.arguments),
                lambda: max_unpooling_1d(symbol_reference, numpy.ones(engine.shape(forward)), **self.arguments)]

    def shape(self, shape_data):
        return pooling_shape(shape_data, dimension=1, **self.arguments)


class MaxReferencePooling2D(Operator):
    def __init__(self, size: tuple, step: tuple):
        self.inputs_count = 2
        self.arguments = {'size': size, 'step': step}

    def compute(self, value_data, reference_data):
        return compute_max_pooling_2d(value_data, reference=reference_data, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_reference):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_2d(symbol_reference, forward, **self.arguments),
                lambda: max_unpooling_2d(symbol_reference, numpy.ones(engine.shape(forward)), **self.arguments)]

    def shape(self, shape_data):
        return pooling_shape(shape_data, dimension=2, **self.arguments)


class MaxUnpooling1D(Operator):
    def __init__(self, size: int, step: int):
        self.inputs_count = 2
        self.arguments = {'size': size, 'step': step}

    def compute(self, value_data, value_pooling):
        return compute_max_unpooling_1d(value_data, value_pooling, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_pooling):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_1d(symbol_data, numpy.ones(engine.shape(symbol_pooling)), **self.arguments),
                lambda: max_pooling_1d(forward, reference=symbol_data, **self.arguments)]

    def shape(self, shape_data, shape_pooling):
        return max_unpooling_shape(shape_data, shape_pooling, 1)


class MaxUnpooling2D(Operator):
    def __init__(self, size: tuple, step: tuple):
        self.inputs_count = 2
        self.arguments = {'size': size, 'step': step}

    def compute(self, value_data, value_pooling):
        return compute_max_unpooling_2d(value_data, value_pooling, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_pooling):
        forward = engine.gradient(symbol_forward)
        return [lambda: max_unpooling_2d(symbol_data, numpy.ones(engine.shape(symbol_pooling)), **self.arguments),
                lambda: max_pooling_2d(forward, reference=symbol_data, **self.arguments)]

    def shape(self, shape_data, shape_pooling):
        return max_unpooling_shape(shape_data, shape_pooling, 2)


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
        return pooling_shape(shape_data, dimension=1, **self.arguments)


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
        return average_unpooling_shape(shape_data, dimension=1, **self.arguments)

