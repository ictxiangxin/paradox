from enum import Enum
from functools import reduce
import numpy
from paradox.kernel.operator import element_wise_shape


class ConvolutionMode(Enum):
    ConvolutionMode = 0
    valid = 1
    full = 2


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
            if key[r] >= array_shape[r]:
                key[r - 1] = key[r] // array_shape[r]
                key[r] %= array_shape[r]
            else:
                break
        yield tuple(key)


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
            for key in __array_key_traversal(final_shape):
                result.append(__compute_convolution_nd(data[key], kernel[key], dimension, mode_string))
            return numpy.array(result).reshape(final_shape + result[0].shape)
        else:
            return __compute_convolution_nd(data, kernel, dimension, mode_string)
    else:
        if data_prefix_shape:
            for data_key in __array_key_traversal(data_prefix_shape):
                if kernel_prefix_shape:
                    for kernel_key in __array_key_traversal(kernel_prefix_shape):
                        result.append(__compute_convolution_nd(data[data_key], kernel[kernel_key], dimension, mode_string))
                else:
                    result.append(__compute_convolution_nd(data[data_key], kernel, dimension, mode_string))
            final_shape = data_prefix_shape + kernel_prefix_shape + basic_convolution_shape(data.shape[-2:], kernel.shape[-2:], 2, mode_string)
            return numpy.array(result).reshape(final_shape)
        else:
            if kernel_prefix_shape:
                for kernel_key in __array_key_traversal(kernel_prefix_shape):
                    result.append(__compute_convolution_nd(data, kernel[kernel_key], dimension, mode_string))
                final_shape = data_prefix_shape + kernel_prefix_shape + basic_convolution_shape(data.shape[-2:], kernel.shape[-2:], 2, mode_string)
                return numpy.array(result).reshape(final_shape)
            else:
                return __compute_convolution_nd(data, kernel, dimension, mode_string)


def __compute_max_pooling_1d(data, size, step, reference=None):
    if data.shape[0] < size:
        raise ValueError('Data shape smaller than pooling size: {} {}'.format(data.shape, size))
    pooling_array = []
    for i in range(0, data.shape[0] - size + 1, step):
        if reference is None:
            pooling_array.append(numpy.max(data[i: i + size]))
        else:
            pooling_array.append(data[i + numpy.argmax(reference[i: i + size])])
    return numpy.array(pooling_array)


def compute_max_pooling_1d(data, size, step, reference=None):
    result = []
    data_prefix_shape = data.shape[:-1]
    if data_prefix_shape:
        for key in __array_key_traversal(data_prefix_shape):
            if reference is None:
                result.append(__compute_max_pooling_1d(data[key], size, step))
            else:
                result.append(__compute_max_pooling_1d(data[key], size, step, reference[key]))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_max_pooling_1d(data, size, step)


def __compute_max_unpooling_1d(data, pooling, size, step):
    if data.shape[0] < size:
        raise ValueError('Data shape smaller than pooling size: {} {}'.format(data.shape, size))
    unpooling_array = numpy.zeros(data.shape)
    for p_i, i in enumerate(range(0, data.shape[0] - size + 1, step)):
        max_index = numpy.argmax(data[i: i + size])
        unpooling_array[i + max_index] += pooling[p_i]
    return unpooling_array


def compute_max_unpooling_1d(data, pooling, size, step):
    result = []
    data_prefix_shape = data.shape[:-1]
    kernel_prefix_shape = pooling.shape[:-1]
    final_shape = element_wise_shape(data_prefix_shape, kernel_prefix_shape)[0]
    data = numpy.broadcast_to(data, final_shape + data.shape[-1:])
    pooling = numpy.broadcast_to(pooling, final_shape + pooling.shape[-1:])
    if final_shape:
        for key in __array_key_traversal(final_shape):
            result.append(__compute_max_unpooling_1d(data[key], pooling[key], size, step))
        return numpy.array(result).reshape(final_shape + result[0].shape)
    else:
        return __compute_max_unpooling_1d(data, pooling, size, step)


def __compute_max_pooling_2d(data, size, step, reference=None):
    if data.shape[0] < size[0] or data.shape[1] < size[1]:
        raise ValueError('Data shape smaller than size: {} {}'.format(data.shape, size))
    pooling_array = []
    for i in range(0, data.shape[0] - size[0] + 1, step[0]):
        for j in range(0, data.shape[1] - size[1] + 1, step[1]):
            if i // step[0] >= len(pooling_array):
                pooling_array.append([])
            if reference is None:
                pooling_array[i // step[0]].append(numpy.max(data[i: i + size[0], j: j + size[1]]))
            else:
                max_index = numpy.argmax(reference[i: i + size[0], j: j + size[1]])
                pooling_array[i // step[0]].append(data[i + max_index // size[1], j + max_index % size[1]])
    return numpy.array(pooling_array)


def compute_max_pooling_2d(data, size, step, reference=None):
    result = []
    data_prefix_shape = data.shape[:-2]
    if data_prefix_shape:
        for key in __array_key_traversal(data_prefix_shape):
            if reference is None:
                result.append(__compute_max_pooling_2d(data[key], size, step))
            else:
                result.append(__compute_max_pooling_2d(data[key], size, step, reference[key]))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_max_pooling_2d(data, size, step)


def __compute_max_unpooling_2d(data, pooling, size, step):
    if data.shape[0] < size[0] or data.shape[1] < size[1]:
        raise ValueError('Data shape smaller than size: {} {}'.format(data.shape, size))
    unpooling_array = numpy.zeros(data.shape)
    for p_i, i in enumerate(range(0, data.shape[0] - size[0] + 1, step[0])):
        for p_j, j in enumerate(range(0, data.shape[1] - size[1] + 1, step[1])):
            max_index = numpy.argmax(data[i: i + size[0], j: j + size[1]])
            unpooling_array[i + max_index // size[1], j + max_index % size[1]] += pooling[p_i, p_j]
    return unpooling_array


def compute_max_unpooling_2d(data, pooling, size, step):
    result = []
    data_prefix_shape = data.shape[:-2]
    kernel_prefix_shape = pooling.shape[:-2]
    final_shape = element_wise_shape(data_prefix_shape, kernel_prefix_shape)[0]
    data = numpy.broadcast_to(data, final_shape + data.shape[-2:])
    pooling = numpy.broadcast_to(pooling, final_shape + pooling.shape[-2:])
    if final_shape:
        for key in __array_key_traversal(final_shape):
            result.append(__compute_max_unpooling_2d(data[key], pooling[key], size, step))
        return numpy.array(result).reshape(final_shape + result[0].shape)
    else:
        return __compute_max_unpooling_2d(data, pooling, size, step)


def __compute_average_pooling_1d(data, size, step):
    if data.shape[0] < size:
        raise ValueError('Data shape smaller than pooling size: {} {}'.format(data.shape, size))
    pooling_array = []
    for i in range(0, data.shape[0] - size + 1, step):
        pooling_array.append(numpy.mean(data[i: i + size]))
    return numpy.array(pooling_array)


def compute_average_pooling_1d(data, size, step):
    result = []
    data_prefix_shape = data.shape[:-1]
    if data_prefix_shape:
        for key in __array_key_traversal(data_prefix_shape):
            result.append(__compute_average_pooling_1d(data[key], size, step))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_average_pooling_1d(data, size, step)


def __compute_average_unpooling_1d(pooling, size, step, unpooling_size=None):
    unpooling_array = numpy.zeros(size + (pooling.shape[0] - 1) * step)
    for p_i, i in enumerate(range(0, unpooling_array.shape[0] - size + 1, step)):
        unpooling_array[i: i + size] += pooling[p_i] * numpy.ones(size)
    if unpooling_size is not None:
        unpooling_array = numpy.concatenate((unpooling_array, numpy.zeros(unpooling_size - unpooling_array.shape[0])))
    return unpooling_array


def compute_average_unpooling_1d(pooling, size, step, unpooling_size=None):
    result = []
    data_prefix_shape = pooling.shape[:-1]
    if data_prefix_shape:
        for key in __array_key_traversal(data_prefix_shape):
            result.append(__compute_average_unpooling_1d(pooling[key], size, step, unpooling_size))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_average_unpooling_1d(pooling, size, step, unpooling_size)


def __compute_average_pooling_2d(data, size, step):
    if data.shape[0] < size[0] or data.shape[1] < size[1]:
        raise ValueError('Data shape smaller than size: {} {}'.format(data.shape, size))
    pooling_array = []
    for i in range(0, data.shape[0] - size[0] + 1, step[0]):
        for j in range(0, data.shape[1] - size[1] + 1, step[1]):
            if i // step[0] >= len(pooling_array):
                pooling_array.append([])
            pooling_array[i // step[0]].append(numpy.mean(data[i: i + size[0], j: j + size[1]]))
    return numpy.array(pooling_array)


def compute_average_pooling_2d(data, size, step):
    result = []
    data_prefix_shape = data.shape[:-2]
    if data_prefix_shape:
        for key in __array_key_traversal(data_prefix_shape):
            result.append(__compute_average_pooling_2d(data[key], size, step))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_average_pooling_2d(data, size, step)


def __compute_average_unpooling_2d(pooling, size, step, unpooling_size=None):
    unpooling_array = numpy.zeros([size[0] + (pooling.shape[0] - 1) * step[0], size[1] + (pooling.shape[1] - 1) * step[1]])
    for p_i, i in enumerate(range(0, unpooling_array.shape[0] - size[0] + 1, step[0])):
        for p_j, j in enumerate(range(0, unpooling_array.shape[1] - size[1] + 1, step[1])):
            unpooling_array[i: i + size[0], j: j + size[1]] += pooling[p_i, p_j] * numpy.ones(size)
    if unpooling_size is not None:
        unpooling_array = numpy.concatenate((unpooling_array, numpy.zeros([unpooling_size[0] - unpooling_array.shape[0], unpooling_array.shape[1]])), axis=0)
        unpooling_array = numpy.concatenate((unpooling_array, numpy.zeros([unpooling_array.shape[0], unpooling_size[1] - unpooling_array.shape[1]])), axis=1)
    return unpooling_array


def compute_average_unpooling_2d(pooling, size, step, unpooling_size=None):
    result = []
    data_prefix_shape = pooling.shape[:-2]
    if data_prefix_shape:
        for key in __array_key_traversal(data_prefix_shape):
            result.append(__compute_average_unpooling_2d(pooling[key], size, step, unpooling_size))
        return numpy.array(result).reshape(data_prefix_shape + result[0].shape)
    else:
        return __compute_average_unpooling_2d(pooling, size, step, unpooling_size)
