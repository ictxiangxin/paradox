import numpy


def xavier_initialization(shape):
    weight = numpy.random.randn(*shape) / numpy.sqrt(shape[0])
    return weight


def he_initialization(shape):
    weight = numpy.random.randn(*shape) / numpy.sqrt(shape[0] / 2)
    return weight
