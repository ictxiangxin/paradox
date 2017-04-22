from enum import Enum
from functools import reduce
import numpy
from paradox.kernel.operator import element_wise_shape
from paradox.utils import array_index_traversal, multi_range


class ConvolutionMode(Enum):
    valid = 0
    full = 1


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


def basic_convolution_shape(shape_data, shape_kernel, dimension: int, mode: str):
    if mode == 'valid':
        return tuple(shape_data[i] - shape_kernel[i] + 1 for i in range(-dimension, 0))
    elif mode == 'full':
        return tuple(shape_data[i] + shape_kernel[i] - 1 for i in range(-dimension, 0))
    else:
        raise ValueError('Invalid convolution mode: {}'.format(mode))


def __compute_valid_convolution_nd(data, kernel, dimension: int):
    convolution_shape = tuple(data.shape[i] - kernel.shape[i] + 1 for i in range(dimension))
    list_dimension = reduce(lambda a, b: a * b, convolution_shape)
    kernel_flat = kernel.ravel()
    data_flat = numpy.zeros((list_dimension, len(kernel_flat)))
    for i in range(list_dimension):
        tensor_slice_start = [0] * len(kernel.shape)
        tensor_slice = [None] * len(kernel.shape)
        tensor_slice_start[-1] = i
        for r in range(len(kernel.shape) - 1, -1, -1):
            dimension_scale = data.shape[r] - kernel.shape[r] + 1
            if tensor_slice_start[r] >= dimension_scale:
                tensor_slice_start[r - 1] = tensor_slice_start[r] // dimension_scale
                tensor_slice_start[r] %= dimension_scale
            tensor_slice[r] = slice(tensor_slice_start[r], tensor_slice_start[r] + kernel.shape[r])
        data_flat[i] = data[tensor_slice].ravel()
    convolution_flat = numpy.matmul(data_flat, numpy.flip(kernel_flat, axis=0))
    convolution_nd = convolution_flat.reshape(convolution_shape)
    return convolution_nd


def __compute_convolution_nd(data, kernel, dimension: int, mode: str):
    mode_string = __get_convolution_mode_string(mode)
    for i in range(dimension):
        if data.shape[i] < kernel.shape[i]:
            raise ValueError('Data shape smaller than kernel shape: {} {}'.format(data.shape, kernel.shape))
    if len(data.shape) == len(kernel.shape) == dimension:
        if mode_string == 'valid':
            return __compute_valid_convolution_nd(data, kernel, dimension)
        elif mode_string == 'full':
            expand_data = numpy.zeros([data.shape[i] + (kernel.shape[i] - 1) * 2 for i in range(dimension)])
            padding = [kernel.shape[i] - 1 for i in range(dimension)]
            expand_data[[slice(padding[i], padding[i] + data.shape[i]) for i in range(dimension)]] = data
            return __compute_valid_convolution_nd(expand_data, kernel, dimension)
        else:
            raise ValueError('Never reached.')
    else:
        raise ValueError('These shapes can not execute convolution-{}d: {} {}'.format(dimension, data.shape, kernel.shape))


def compute_convolution_nd(data, kernel, dimension: int, mode=ConvolutionMode.valid, element_wise: bool=False):
    mode_string = __get_convolution_mode_string(mode)
    result = []
    data_prefix_shape = data.shape[:-2]
    kernel_prefix_shape = kernel.shape[:-2]
    if element_wise:
        final_shape = element_wise_shape(data_prefix_shape, kernel_prefix_shape)[0]
        data = numpy.broadcast_to(data, final_shape + data.shape[-2:])
        kernel = numpy.broadcast_to(kernel, final_shape + kernel.shape[-2:])
        if final_shape:
            for index in array_index_traversal(final_shape):
                result.append(__compute_convolution_nd(data[index], kernel[index], dimension, mode_string))
            return numpy.array(result).reshape(final_shape + result[0].shape)
        else:
            return __compute_convolution_nd(data, kernel, dimension, mode_string)
    else:
        if data_prefix_shape:
            for data_index in array_index_traversal(data_prefix_shape):
                if kernel_prefix_shape:
                    for kernel_index in array_index_traversal(kernel_prefix_shape):
                        result.append(__compute_convolution_nd(data[data_index], kernel[kernel_index], dimension, mode_string))
                else:
                    result.append(__compute_convolution_nd(data[data_index], kernel, dimension, mode_string))
            final_shape = data_prefix_shape + kernel_prefix_shape + basic_convolution_shape(data.shape[-2:], kernel.shape[-2:], 2, mode_string)
            return numpy.array(result).reshape(final_shape)
        else:
            if kernel_prefix_shape:
                for kernel_index in array_index_traversal(kernel_prefix_shape):
                    result.append(__compute_convolution_nd(data, kernel[kernel_index], dimension, mode_string))
                final_shape = data_prefix_shape + kernel_prefix_shape + basic_convolution_shape(data.shape[-2:], kernel.shape[-2:], 2, mode_string)
                return numpy.array(result).reshape(final_shape)
            else:
                return __compute_convolution_nd(data, kernel, dimension, mode_string)


def __compute_max_pooling_nd(data, size, step, dimension: int, reference=None):
    for i in range(dimension):
        if data.shape[i] < size[i]:
            raise ValueError('Data shape smaller than size: {} {}'.format(data.shape, size))
    pooling_array = []
    pooling_grid = [range(0, data.shape[i] - size[i] + 1, step[i]) for i in range(dimension)]
    for index in multi_range(pooling_grid):
        sub_slice = [slice(index[i], index[i] + size[i]) for i in range(dimension)]
        if reference is None:
            pooling_array.append(numpy.max(data[sub_slice]))
        else:
            max_index = numpy.argmax(reference[sub_slice])
            sub_data = data[sub_slice]
            pooling_array.append(sub_data[numpy.unravel_index(max_index, sub_data.shape)])
    return numpy.array(pooling_array).reshape([len(g) for g in pooling_grid])


def compute_max_pooling_nd(data, size, step, dimension: int, reference=None):
    result = []
    data_prefix_shape = data.shape[:-dimension]
    if data_prefix_shape:
        for key in array_index_traversal(data_prefix_shape):
            if reference is None:
                result.append(__compute_max_pooling_nd(data[key], size, step, dimension))
            else:
                result.append(__compute_max_pooling_nd(data[key], size, step, dimension, reference[key]))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_max_pooling_nd(data, size, step, dimension)


def __compute_max_unpooling_nd(data, pooling, size, step, dimension: int):
    for i in range(dimension):
        if data.shape[i] < size[i]:
            raise ValueError('Data shape smaller than size: {} {}'.format(data.shape, size))
    unpooling_array = numpy.zeros(data.shape)
    unpooling_grid = [range(0, data.shape[i] - size[i] + 1, step[i]) for i in range(dimension)]
    for n, index in enumerate(multi_range(unpooling_grid)):
        sub_slice = [slice(index[i], index[i] + size[i]) for i in range(dimension)]
        max_index = numpy.argmax(data[sub_slice])
        sub_unpooling_array = unpooling_array[sub_slice]
        sub_unpooling_array[numpy.unravel_index(max_index, sub_unpooling_array.shape)] = pooling[numpy.unravel_index(n, pooling.shape)]
    return unpooling_array


def compute_max_unpooling_nd(data, pooling, size, step, dimension: int):
    result = []
    data_prefix_shape = data.shape[:-dimension]
    kernel_prefix_shape = pooling.shape[:-dimension]
    final_shape = element_wise_shape(data_prefix_shape, kernel_prefix_shape)[0]
    data = numpy.broadcast_to(data, final_shape + data.shape[-dimension:])
    pooling = numpy.broadcast_to(pooling, final_shape + pooling.shape[-dimension:])
    if final_shape:
        for key in array_index_traversal(final_shape):
            result.append(__compute_max_unpooling_nd(data[key], pooling[key], size, step, dimension))
        return numpy.array(result).reshape(final_shape + result[0].shape)
    else:
        return __compute_max_unpooling_nd(data, pooling, size, step, dimension)


def __compute_average_pooling_nd(data, size, step, dimension: int):
    for i in range(dimension):
        if data.shape[i] < size[i]:
            raise ValueError('Data shape smaller than size: {} {}'.format(data.shape, size))
    pooling_array = []
    pooling_grid = [range(0, data.shape[i] - size[i] + 1, step[i]) for i in range(dimension)]
    for index in multi_range(pooling_grid):
        pooling_array.append(numpy.mean(data[[slice(index[i], index[i] + size[i]) for i in range(dimension)]]))
    return numpy.array(pooling_array).reshape([len(g) for g in pooling_grid])


def compute_average_pooling_nd(data, size, step, dimension: int):
    result = []
    data_prefix_shape = data.shape[:-dimension]
    if data_prefix_shape:
        for key in array_index_traversal(data_prefix_shape):
            result.append(__compute_average_pooling_nd(data[key], size, step, dimension))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_average_pooling_nd(data, size, step, dimension)


def __compute_average_unpooling_nd(pooling, size, step, dimension: int, unpooling_size=None):
    if unpooling_size is None:
        unpooling_array = numpy.zeros([size[i] + (pooling.shape[i] - 1) * step[i] for i in range(dimension)])
    else:
        unpooling_array = numpy.zeros(unpooling_size)
    unpooling_grid = [range(0, unpooling_array.shape[i] - size[i] + 1, step[i]) for i in range(dimension)]
    for n, index in enumerate(multi_range(unpooling_grid)):
        sub_slice = [slice(index[i], index[i] + size[i]) for i in range(dimension)]
        unpooling_array[sub_slice] += pooling[numpy.unravel_index(n, pooling.shape)]
    return unpooling_array


def compute_average_unpooling_nd(pooling, size, step, dimension: int, unpooling_size=None):
    result = []
    data_prefix_shape = pooling.shape[:-dimension]
    if data_prefix_shape:
        for key in array_index_traversal(data_prefix_shape):
            result.append(__compute_average_unpooling_nd(pooling[key], size, step, dimension, unpooling_size))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_average_unpooling_nd(pooling, size, step, dimension, unpooling_size)
