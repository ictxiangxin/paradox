from paradox.kernel.operator import Operator, element_wise_shape
from paradox.neural_network.convolutional_neural_network.compute import ConvolutionMode, compute_convolution_1d
from paradox.neural_network.convolutional_neural_network.function import convolution_1d


class Convolution1D(Operator):
    def __init__(self, mode):
        self.arguments = {'mode': mode}

    def compute(self, value_data, value_kernel):
        return compute_convolution_1d(value_data, value_kernel, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_data, symbol_kernel):
        forward = engine.gradient(symbol_forward)
        return [lambda: convolution_1d(forward, symbol_kernel[::-1], ConvolutionMode.full),
                lambda: convolution_1d(symbol_data, forward, ConvolutionMode.valid)]

    def shape(self, shape_data, shape_kernel):
        prefix_shape, prefix_broadcast_data, prefix_broadcast_kernel = element_wise_shape(shape_data[:-1], shape_kernel[:-1])
        mode = self.arguments['mode']
        if mode == 'valid' or mode == ConvolutionMode.valid:
            new_shape = prefix_shape + (shape_data[-1] - shape_kernel[-1] + 1,)
        elif mode == 'same' or mode == ConvolutionMode.same:
            new_shape = prefix_shape + (shape_data[-1],)
        elif mode == 'full' or mode == ConvolutionMode.full:
            new_shape = prefix_shape + (shape_data[-1] + shape_kernel[-1] - 1,)
        else:
            raise ValueError('Invalid convolution mode: {}'.format(mode))
        return new_shape, prefix_broadcast_data + (0,), prefix_broadcast_kernel+(0,)
