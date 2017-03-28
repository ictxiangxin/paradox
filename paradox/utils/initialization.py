import numpy


def xavier_initialization(shape):
    matrix = numpy.random.randn(*shape) / numpy.sqrt(shape[0])
    return matrix


def he_initialization(shape):
    matrix = numpy.random.randn(*shape) / numpy.sqrt(shape[0] / 2)
    return matrix


def normal_initialization(shape):
    matrix = numpy.random.normal(0, 1, shape)
    return matrix


def uniform_initialization(shape):
    matrix = numpy.random.randn(*shape)
    return matrix
