from enum import Enum
from functools import reduce
import numpy
from paradox.kernel.operator import element_wise_shape


ConvolutionMode = Enum('ConvolutionMode', ('valid', 'full'))


convolution_map = {
    'valid': ConvolutionMode.valid,
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


def __array_key_traversal(array_shape):
    scale = reduce(lambda a, b: a * b, array_shape)
    for i in range(scale):
        key = [0] * len(array_shape)
        key[-1] = i
        for r in range(len(array_shape) - 1, -1, -1):
            if key[r] > array_shape[r]:
                key[r - 1] = key[r] // array_shape[r]
                key[r] %= array_shape[r]
            else:
                break
        yield tuple(key)


def __compute_valid_convolution_2d(data, kernel):
    convolution_result = numpy.zeros((data.shape[0] - kernel.shape[0] + 1, data.shape[1] - kernel.shape[1] + 1))
    for i in range(kernel.shape[0], data.shape[0] + 1):
        for j in range(kernel.shape[1], data.shape[1] + 1):
            sub_data = data[i - kernel.shape[0]: i, j - kernel.shape[1]: j]
            convolution_value = numpy.convolve(sub_data.ravel(), kernel.ravel(), 'valid')[0]
            convolution_result[i - kernel.shape[0], j - kernel.shape[1]] = convolution_value
    return convolution_result


def __compute_convolution_1d(data, kernel, mode: str):
    if data.shape[0] < kernel.shape[0]:
        raise ValueError('Kernel shape smaller than data shape: {} {}'.format(data.shape, kernel.shape))
    if len(data.shape) == len(kernel.shape) == 1:
        return numpy.convolve(data, kernel, mode)
    else:
        raise ValueError('These shapes can not execute convolve-1d: {} {}'.format(data.shape, kernel.shape))


def compute_convolution_1d(data, kernel, mode=ConvolutionMode.valid):
    mode_string = __get_convolution_mode_string(mode)
    result = []
    data_prefix_shape = data.shape[:-1]
    kernel_prefix_shape = kernel.shape[:-1]
    final_shape = element_wise_shape(data_prefix_shape, kernel_prefix_shape)[0]
    data = numpy.broadcast_to(data, final_shape + (data.shape[-1],))
    kernel = numpy.broadcast_to(kernel, final_shape + (kernel.shape[-1],))
    if final_shape:
        for key in __array_key_traversal(final_shape):
            result.append(__compute_convolution_1d(data[key].ravel(), kernel[key].ravel(), mode_string))
        return numpy.array(result).reshape(final_shape + result[0].shape)
    else:
        return __compute_convolution_1d(data, kernel, mode_string)


def __compute_convolution_2d(data, kernel, mode: str):
    mode_string = __get_convolution_mode_string(mode)
    if data.shape[0] < kernel.shape[0] or data.shape[1] < kernel.shape[1]:
        raise ValueError('Kernel shape smaller than data shape: {} {}'.format(data.shape, kernel.shape))
    if len(data.shape) == len(kernel.shape) == 2:
        if mode_string == 'valid':
            return __compute_valid_convolution_2d(data, kernel)
        elif mode_string == 'full':
            expand_data = numpy.zeros((data.shape[0] + (kernel.shape[0] - 1) * 2, data.shape[1] + (kernel.shape[1] - 1) * 2))
            x_padding = kernel.shape[0] - 1
            y_padding = kernel.shape[1] - 1
            expand_data[x_padding: x_padding + data.shape[0], y_padding: y_padding + data.shape[1]] = data
            return __compute_valid_convolution_2d(expand_data, kernel)
        else:
            raise ValueError('Never reached.')
    else:
        raise ValueError('These shapes can not execute convolve-2d: {} {}'.format(data.shape, kernel.shape))


def compute_convolution_2d(data, kernel, mode=ConvolutionMode.valid):
    mode_string = __get_convolution_mode_string(mode)
    result = []
    data_prefix_shape = data.shape[:-2]
    kernel_prefix_shape = kernel.shape[:-2]
    final_shape = element_wise_shape(data_prefix_shape, kernel_prefix_shape)[0]
    data = numpy.broadcast_to(data, final_shape + data.shape[-2:])
    kernel = numpy.broadcast_to(kernel, final_shape + kernel.shape[-2:])
    if final_shape:
        for key in __array_key_traversal(final_shape):
            result.append(__compute_convolution_2d(data[key], kernel[key], mode_string))
        return numpy.array(result).reshape(final_shape + result[0].shape)
    else:
        return __compute_convolution_2d(data, kernel, mode_string)
