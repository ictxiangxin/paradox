from abc import abstractmethod
import numpy
from paradox.kernel.symbol import Variable


class ConnectionLayer:
    @abstractmethod
    def weight_bias(self, input_dimension):
        pass

    @abstractmethod
    def input_dimension(self):
        pass

    @abstractmethod
    def output_dimension(self):
        pass


class Dense(ConnectionLayer):
    def __init__(self, output_dimension: int, input_dimension: int=None):
        self.__input_dimension = input_dimension
        self.__output_dimension = output_dimension

    def weight_bias(self, input_dimension: int=None):
        if self.__input_dimension is None:
            if input_dimension is None:
                raise ValueError('Need input dimension.')
            else:
                self.__input_dimension = input_dimension
        else:
            if input_dimension is not None:
                if self.__input_dimension != input_dimension:
                    raise ValueError('Not match input dimension: {} {}'.format(self.__input_dimension, input_dimension))
        weight = Variable(numpy.zeros((self.__output_dimension, self.__input_dimension)))
        bias = Variable(numpy.zeros((self.__output_dimension, 1)))
        return weight, bias

    def input_dimension(self):
        return self.__input_dimension

    def output_dimension(self):
        return self.__output_dimension


connection_map = {
    'dense': Dense,
}


def register_connection(name: str, connection: ConnectionLayer):
    connection_map[name.lower()] = connection


class Connection:
    def __init__(self, name: str):
        self.__name = name.lower()
        self.__connection = None
        if self.__name in connection_map:
            self.__connection = connection_map[self.__name]
        else:
            raise ValueError('No such connection: {}'.format(name))

    def connection_layer(self):
        return self.__connection
