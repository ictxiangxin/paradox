from paradox.kernel import flip, rotate90
from paradox.kernel.operator import Operator, element_wise_shape
from paradox.neural_network.convolutional_neural_network.compute import \
    ConvolutionMode, \
    compute_convolution_1d, \
    compute_convolution_2d
from paradox.neural_network.convolutional_neural_network.function import \
    convolution_1d, \
    convolution_2d


def convolution_shape(shape_data, shape_kernel, mode, dimension):
    prefix_shape, prefix_broadcast_data, prefix_broadcast_kernel = element_wise_shape(shape_data[:-dimension], shape_kernel[:-dimension])
    if mode == 'valid' or mode == ConvolutionMode.valid:
        new_shape = prefix_shape + tuple(shape_data[i] - shape_kernel[i] + 1 for i in range(-dimension, 0))
    elif mode == 'full' or mode == ConvolutionMode.full:
        new_shape = prefix_shape + tuple(shape_data[i] + shape_kernel[i] - 1 for i in range(-dimension, 0))
    else:
        raise ValueError('Invalid convolution mode: {}'.format(mode))
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
