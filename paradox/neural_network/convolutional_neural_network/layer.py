from abc import abstractmethod
import numpy
from paradox.kernel.symbol import Variable
from paradox.neural_network.convolutional_neural_network.function import \
    convolution_1d, \
    convolution_2d, \
    max_pooling_1d, \
    max_pooling_2d, \
    max_unpooling_1d, \
    max_unpooling_2d, \
    average_pooling_1d, \
    average_pooling_2d, \
    average_unpooling_1d, \
    average_unpooling_2d


class ConvolutionLayer:
    def __init__(self, kernel_shape, mode):
        self.__kernel_shape = kernel_shape
        self.__mode = mode
        self.__kernel = Variable(numpy.random.normal(0, 1, self.__kernel_shape))

    def kernel(self):
        return self.__kernel

    def mode(self):
        return self.__mode

    @abstractmethod
    def convolution_function(self):
        pass


class Convolution1DLayer(ConvolutionLayer):
    def convolution_function(self):
        return convolution_1d


class Convolution2DLayer(ConvolutionLayer):
    def convolution_function(self):
        return convolution_2d


convolution_map = {
    '1d': Convolution1DLayer,
    '2d': Convolution2DLayer,
}


def register_convolution(name: str, convolution: ConvolutionLayer):
    convolution_map[name.lower()] = convolution


class Convolution:
    def __init__(self, name: str, *args, **kwargs):
        self.__name = name.lower()
        self.__convolution = None
        if self.__name in convolution_map:
            self.__convolution = convolution_map[self.__name](*args, **kwargs)
        else:
            raise ValueError('No such convolution: {}'.format(name))

    def convolution_layer(self):
        return self.__convolution


class PoolingLayer:
    def __init__(self, size, step):
        self.__size = size
        self.__step = step

    def size(self):
        return self.__size

    def step(self):
        return self.__step

    @abstractmethod
    def pooling_function(self):
        pass


class MaxPooling1DLayer(PoolingLayer):
    def pooling_function(self):
        return max_pooling_1d


class MaxPooling2DLayer(PoolingLayer):
    def pooling_function(self):
        return max_pooling_2d


class AveragePooling1DLayer(PoolingLayer):
    def pooling_function(self):
        return average_pooling_1d


class AveragePooling2DLayer(PoolingLayer):
    def pooling_function(self):
        return average_pooling_2d


pooling_map = {
    'max1d': MaxPooling1DLayer,
    'max2d': MaxPooling2DLayer,
    'average1d': AveragePooling1DLayer,
    'average2d': AveragePooling2DLayer,
}


def register_pooling(name: str, pooling: PoolingLayer):
    pooling_map[name.lower()] = pooling


class Pooling:
    def __init__(self, name: str, *args, **kwargs):
        self.__name = name.lower()
        self.__pooling = None
        if self.__name in pooling_map:
            self.__pooling = pooling_map[self.__name](*args, **kwargs)
        else:
            raise ValueError('No such pooling: {}'.format(name))

    def pooling_layer(self):
        return self.__pooling


class UnpoolingLayer:
    def __init__(self, size, step):
        self.__size = size
        self.__step = step

    def size(self):
        return self.__size

    def step(self):
        return self.__step

    @abstractmethod
    def unpooling_function(self):
        pass


class MaxUnpooling1DLayer(UnpoolingLayer):
    def unpooling_function(self):
        return max_unpooling_1d


class MaxUnpooling2DLayer(UnpoolingLayer):
    def unpooling_function(self):
        return max_unpooling_2d


class AverageUnpooling1DLayer(UnpoolingLayer):
    def unpooling_function(self):
        return average_unpooling_1d


class AverageUnpooling2DLayer(UnpoolingLayer):
    def unpooling_function(self):
        return average_unpooling_2d


unpooling_map = {
    'max1d': MaxUnpooling1DLayer,
    'max2d': MaxUnpooling2DLayer,
    'average1d': AverageUnpooling1DLayer,
    'average2d': AverageUnpooling2DLayer,
}


def register_unpooling(name: str, unpooling: UnpoolingLayer):
    unpooling_map[name.lower()] = unpooling


class Unpooling:
    def __init__(self, name: str, *args, **kwargs):
        self.__name = name.lower()
        self.__unpooling = None
        if self.__name in unpooling_map:
            self.__unpooling = unpooling_map[self.__name](*args, **kwargs)
        else:
            raise ValueError('No such unpooling: {}'.format(name))

    def unpooling_layer(self):
        return self.__unpooling
