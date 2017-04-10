from abc import abstractmethod
import numpy
from paradox.kernel.symbol import Variable


class ConnectionLayer:
    input_dimension = None
    output_dimension = None

    def __init__(self, output_dimension: int, input_dimension: int=None):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def set_input_dimension(self, input_dimension: int=None):
        self.input_dimension = input_dimension

    @abstractmethod
    def weight_bias(self):
        pass


class Dense(ConnectionLayer):
    def weight_bias(self):
        weight = Variable(numpy.zeros((self.input_dimension, self.output_dimension)))
        bias = Variable(numpy.zeros((1, self.output_dimension)))
        return weight, bias


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
