from abc import abstractmethod
import numpy
from paradox.kernel.symbol import Variable
from paradox.neural_network.convolutional_neural_network.function import \
    convolution_nd, \
    max_pooling_nd, \
    max_unpooling_nd, \
    average_pooling_nd, \
    average_unpooling_nd
from paradox.neural_network.convolutional_neural_network.operator import \
    convolution_nd_shape, \
    pooling_nd_shape, \
    unpooling_nd_shape


class ConvolutionLayer:
    kernel_shape = None
    mode = None
    input_shape = None

    def __init__(self, kernel_shape, mode, input_shape=None):
        self.kernel_shape = kernel_shape
        self.mode = mode
        self.input_shape = input_shape
        self._kernel = None

    def convolution_layer(self):
        return self

    def kernel(self):
        if self._kernel is None:
            self._kernel = Variable(numpy.random.normal(0, 1, self.kernel_shape))
        return self._kernel

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def convolution_function(self):
        pass


class ConvolutionNDLayer(ConvolutionLayer):
    def __init__(self, kernel_shape: tuple, mode, dimension: int, input_shape: tuple=None):
        ConvolutionLayer.__init__(self, kernel_shape, mode, input_shape)
        self.__dimension = dimension

    def convolution_function(self):
        return lambda data, kernel, mode: convolution_nd(data, kernel, self.__dimension, mode)

    def get_output_shape(self):
        return convolution_nd_shape(self.input_shape, self.kernel_shape, self.__dimension, self.mode)[0]


class Convolution1DLayer(ConvolutionNDLayer):
    def __init__(self, kernel_shape, mode, input_shape=None):
        ConvolutionNDLayer.__init__(self, kernel_shape, mode, 1, input_shape)


class Convolution2DLayer(ConvolutionNDLayer):
    def __init__(self, kernel_shape, mode, input_shape=None):
        ConvolutionNDLayer.__init__(self, kernel_shape, mode, 2, input_shape)


class Convolution3DLayer(ConvolutionNDLayer):
    def __init__(self, kernel_shape, mode, input_shape=None):
        ConvolutionNDLayer.__init__(self, kernel_shape, mode, 3, input_shape)


convolution_map = {
    'nd': ConvolutionNDLayer,
    '1d': Convolution1DLayer,
    '2d': Convolution2DLayer,
    '3d': Convolution2DLayer,
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
    size = None
    step = None
    input_shape = None

    def __init__(self, size: tuple, step: tuple, input_shape: tuple=None):
        self.size = size
        self.step = step
        self.input_shape = input_shape

    def pooling_layer(self):
        return self

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def pooling_function(self):
        pass


class MaxPoolingNDLayer(PoolingLayer):
    def __init__(self, size: tuple, step: tuple, dimension: int, input_shape: tuple=None):
        PoolingLayer.__init__(self, size, step, input_shape)
        self.__dimension = dimension

    def pooling_function(self):
        return lambda data, size, step: max_pooling_nd(data, size, step, self.__dimension)

    def get_output_shape(self):
        return pooling_nd_shape(self.input_shape, self.size, self.step, self.__dimension)[0]


class MaxPooling1DLayer(MaxPoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple=None):
        MaxPoolingNDLayer.__init__(self, size, step, 1, input_shape)


class MaxPooling2DLayer(MaxPoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple=None):
        MaxPoolingNDLayer.__init__(self, size, step, 2, input_shape)


class MaxPooling3DLayer(MaxPoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple=None):
        MaxPoolingNDLayer.__init__(self, size, step, 3, input_shape)


class AveragePoolingNDLayer(PoolingLayer):
    def __init__(self, size: tuple, step: tuple, dimension: int, input_shape: tuple=None):
        PoolingLayer.__init__(self, size, step, input_shape)
        self.__dimension = dimension

    def pooling_function(self):
        return lambda data, size, step: average_pooling_nd(data, size, step, self.__dimension)

    def get_output_shape(self):
        return pooling_nd_shape(self.input_shape, self.size, self.step, self.__dimension)[0]


class AveragePooling1DLayer(AveragePoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple = None):
        AveragePoolingNDLayer.__init__(self, size, step, 1, input_shape)


class AveragePooling2DLayer(AveragePoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple = None):
        AveragePoolingNDLayer.__init__(self, size, step, 2, input_shape)


class AveragePooling3DLayer(AveragePoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple = None):
        AveragePoolingNDLayer.__init__(self, size, step, 3, input_shape)


pooling_map = {
    'max_nd': MaxPoolingNDLayer,
    'max_1d': MaxPooling1DLayer,
    'max_2d': MaxPooling2DLayer,
    'max_3d': MaxPooling3DLayer,
    'average_nd': AveragePoolingNDLayer,
    'average_1d': AveragePooling1DLayer,
    'average_2d': AveragePooling2DLayer,
    'average_3d': AveragePooling3DLayer,
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
    def __init__(self, size: tuple, step: tuple, input_shape: tuple=None):
        self.size = size
        self.step = step
        self.input_shape = input_shape

    def unpooling_layer(self):
        return self

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def unpooling_function(self):
        pass


class AverageUnpoolingNDLayer(UnpoolingLayer):
    def __init__(self, size: tuple, step: tuple, dimension: int, input_shape: tuple=None):
        UnpoolingLayer.__init__(self, size, step, input_shape)
        self.__dimension = dimension

    def unpooling_function(self):
        return lambda data, pooling, size, step: average_unpooling_nd(pooling, size, step, self.__dimension)

    def get_output_shape(self):
        return unpooling_nd_shape(self.input_shape, self.size, self.step, None, self.__dimension)[0]


class AverageUnpooling1DLayer(AverageUnpoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple=None):
        AverageUnpoolingNDLayer.__init__(self, size, step, 1, input_shape)


class AverageUnpooling2DLayer(AverageUnpoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple=None):
        AverageUnpoolingNDLayer.__init__(self, size, step, 2, input_shape)


class AverageUnpooling3DLayer(AverageUnpoolingNDLayer):
    def __init__(self, size: tuple, step: tuple, input_shape: tuple=None):
        AverageUnpoolingNDLayer.__init__(self, size, step, 3, input_shape)


unpooling_map = {
    'average_nd': AverageUnpoolingNDLayer,
    'average_1d': AverageUnpooling1DLayer,
    'average_2d': AverageUnpooling2DLayer,
    'average_3d': AverageUnpooling3DLayer,
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
