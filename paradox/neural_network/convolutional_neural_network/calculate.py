from enum import Enum
import numpy


ConvolutionMode = Enum('ConvolutionMode', ('valid', 'same', 'full'))


convolution_map = {
    'valid': ConvolutionMode.valid,
    'same': ConvolutionMode.same,
    'full': ConvolutionMode.full,
}


def __get_convolution_mode_string(mode):
    if isinstance(mode, str):
        if mode in convolution_map:
            return mode
        else:
            raise ValueError('No such convolution mode: {}'.format(mode))
    elif isinstance(mode, ConvolutionMode):
        return mode.name
    else:
        raise ValueError('Invalid mode type: {}'.format(type(mode)))


def __valid_convolution(data: numpy.ndarray, kernel: numpy.ndarray):
    convolution_result = numpy.zeros((data.shape[0] - kernel.shape[0] + 1, data.shape[1] - kernel.shape[1] + 1))
    for i in range(kernel.shape[0], data.shape[0] + 1):
        for j in range(kernel.shape[1], data.shape[1] + 1):
            sub_data = data[i - kernel.shape[0]: i, j - kernel.shape[1]: j]
            convolution_value = numpy.convolve(sub_data.ravel(), kernel.ravel(), 'valid')[0]
            convolution_result[i - kernel.shape[0], j - kernel.shape[1]] = convolution_value
    return convolution_result


def convolution_1d(data: numpy.ndarray, kernel: numpy.ndarray, mode=ConvolutionMode.full):
    mode_string = __get_convolution_mode_string(mode)
    if data.shape[0] < kernel.shape[0]:
        raise ValueError('Kernel shape smaller than data shape: {} {}'.format(data.shape, kernel.shape))
    if len(data.shape) == len(kernel.shape) == 1:
        return numpy.convolve(data, kernel, mode_string)
    else:
        raise ValueError('These shapes can not execute convolve-1d: {} {}'.format(data.shape, kernel.shape))


def convolution_2d(data, kernel, mode=ConvolutionMode.full):
    mode_string = __get_convolution_mode_string(mode)
    if data.shape[0] < kernel.shape[0] or data.shape[1] < kernel.shape[1]:
        raise ValueError('Kernel shape smaller than data shape: {} {}'.format(data.shape, kernel.shape))
    if len(data.shape) == len(kernel.shape) == 2:
        if mode_string == 'valid':
            return __valid_convolution(data, kernel)
        elif mode_string == 'same':
            expand_data = numpy.zeros((data.shape[0] + kernel.shape[0] - 1, data.shape[1] + kernel.shape[1] - 1))
            x_padding = kernel.shape[0] // 2
            y_padding = kernel.shape[1] // 2
            expand_data[x_padding: x_padding + data.shape[0], y_padding: y_padding + data.shape[1]] = data
            return __valid_convolution(expand_data, kernel)
        elif mode_string == 'full':
            expand_data = numpy.zeros((data.shape[0] + (kernel.shape[0] - 1) * 2, data.shape[1] + (kernel.shape[1] - 1) * 2))
            x_padding = kernel.shape[0] - 1
            y_padding = kernel.shape[1] - 1
            expand_data[x_padding: x_padding + data.shape[0], y_padding: y_padding + data.shape[1]] = data
            return __valid_convolution(expand_data, kernel)
        else:
            raise ValueError('Never reached.')
    else:
        raise ValueError('These shapes can not execute convolve-2d: {} {}'.format(data.shape, kernel.shape))
