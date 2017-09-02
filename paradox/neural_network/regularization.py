from abc import abstractmethod
from paradox.kernel import *


class RegularizationLayer:
    @abstractmethod
    def regularization_term(self, *args, **kwargs):
        pass


class RegularizationL1(RegularizationLayer):
    @staticmethod
    def regularization_term(target_symbol: Symbol, decay: float):
        return decay * reduce_sum(absolute(target_symbol))


class RegularizationL2(RegularizationLayer):
    @staticmethod
    def regularization_term(target_symbol: Symbol, decay: float):
        return decay * reduce_sum(target_symbol ** 2) ** 0.5


regularization_l1 = RegularizationL1.regularization_term
regularization_l2 = RegularizationL2.regularization_term


regularization_map = {
    'l1': RegularizationL1,
    'l2': RegularizationL2,
}


def register_regularization(name: str, regularization: RegularizationLayer):
    regularization_map[name.lower()] = regularization


class Regularization:
    def __init__(self, name: str, *args, **kwargs):
        self.__name = name.lower()
        self.__regularization = None
        if self.__name in regularization_map:
            self.__regularization = regularization_map[self.__name](*args, **kwargs)
        else:
            raise ValueError('No such regularization: {}'.format(name))

    def regularization_layer(self):
        return self.__regularization
