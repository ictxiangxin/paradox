from abc import abstractmethod
import numpy
from paradox.kernel.symbol import Variable


class ConnectionLayer:
    input_dimension = None
    output_dimension = None

    def __init__(self, output_dimension: int, input_dimension: int=None):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weight = None
        self.bias = None

    @abstractmethod
    def weight_bias(self):
        pass


class Dense(ConnectionLayer):
    def weight_bias(self):
        if self.weight is None:
            self.weight = Variable(numpy.zeros((self.input_dimension, self.output_dimension)))
        if self.bias is None:
            self.bias = Variable(numpy.zeros((1, self.output_dimension)))
        return self.weight, self.bias


connection_map = {
    'dense': Dense,
}


def register_connection(name: str, connection: ConnectionLayer):
    connection_map[name.lower()] = connection


class Connection:
    def __init__(self, name: str, *args, **kwargs):
        self.__name = name.lower()
        self.__connection = None
        if self.__name in connection_map:
            self.__connection = connection_map[self.__name](*args, **kwargs)
        else:
            raise ValueError('No such connection: {}'.format(name))

    def connection_layer(self):
        return self.__connection
