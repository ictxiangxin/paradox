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
from paradox.neural_network.convolutional_neural_network.operator import \
    convolution_nd_shape, \
    pooling_shape, \
    unpooling_shape


class ConvolutionLayer:
    kernel_shape = None
    mode = None
    input_shape = None
    kernel = None

    def __init__(self, kernel_shape, mode, input_shape=None):
        self.kernel_shape = kernel_shape
        self.mode = mode
        self.input_shape = input_shape
        self.kernel = Variable(numpy.random.normal(0, 1, self.kernel_shape))

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def convolution_function(self):
        pass


class Convolution1DLayer(ConvolutionLayer):
    def __init__(self, kernel_shape, mode, input_shape=None):
        ConvolutionLayer.__init__(self, kernel_shape, mode, input_shape)

    def convolution_function(self):
        return convolution_1d

    def get_output_shape(self):
        return convolution_nd_shape(self.input_shape, self.kernel_shape, 1, self.mode)[0]


class Convolution2DLayer(ConvolutionLayer):
    def __init__(self, kernel_shape, mode, input_shape=None):
        ConvolutionLayer.__init__(self, kernel_shape, mode, input_shape)

    def convolution_function(self):
        return convolution_2d

    def get_output_shape(self):
        return convolution_nd_shape(self.input_shape, self.kernel_shape, 2, self.mode)[0]


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
    size = None
    step = None
    input_shape = None

    def __init__(self, size, step, input_shape=None):
        self.size = size
        self.step = step
        self.input_shape = input_shape

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def pooling_function(self):
        pass


class MaxPooling1DLayer(PoolingLayer):
    def pooling_function(self):
        return max_pooling_1d

    def get_output_shape(self):
        return pooling_shape(self.input_shape, self.size, self.step, 1)[0]


class MaxPooling2DLayer(PoolingLayer):
    def pooling_function(self):
        return max_pooling_2d

    def get_output_shape(self):
        return pooling_shape(self.input_shape, self.size, self.step, 2)[0]


class AveragePooling1DLayer(PoolingLayer):
    def pooling_function(self):
        return average_pooling_1d

    def get_output_shape(self):
        return pooling_shape(self.input_shape, self.size, self.step, 1)[0]


class AveragePooling2DLayer(PoolingLayer):
    def pooling_function(self):
        return average_pooling_2d

    def get_output_shape(self):
        return pooling_shape(self.input_shape, self.size, self.step, 2)[0]


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
    def __init__(self, size, step, input_shape=None):
        self.size = size
        self.step = step
        self.input_shape = input_shape

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def unpooling_function(self):
        pass


class MaxUnpooling1DLayer(UnpoolingLayer):
    def unpooling_function(self):
        return max_unpooling_1d

    def get_output_shape(self):
        return unpooling_shape(self.input_shape, self.size, self.step, None, 1)[0]


class MaxUnpooling2DLayer(UnpoolingLayer):
    def unpooling_function(self):
        return max_unpooling_2d

    def get_output_shape(self):
        return unpooling_shape(self.input_shape, self.size, self.step, None, 2)[0]


class AverageUnpooling1DLayer(UnpoolingLayer):
    def unpooling_function(self):
        return average_unpooling_1d

    def get_output_shape(self):
        return unpooling_shape(self.input_shape, self.size, self.step, None, 2)[0]


class AverageUnpooling2DLayer(UnpoolingLayer):
    def unpooling_function(self):
        return average_unpooling_2d

    def get_output_shape(self):
        return unpooling_shape(self.input_shape, self.size, self.step, None, 2)[0]


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
