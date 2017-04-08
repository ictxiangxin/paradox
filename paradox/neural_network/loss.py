from enum import Enum
from abc import abstractmethod
from paradox.kernel import *
from paradox.utils import generate_class_matrix


LossCategory = Enum('LossCategory', ('classification', 'regression'))


class LossLayer:
    loss_type = None

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        pass


class SoftMaxLoss(LossLayer):
    loss_type = LossCategory.classification

    @staticmethod
    def loss_function(input_symbol: Symbol, classification):
        class_matrix = generate_class_matrix(classification)
        class_symbol = Symbol(class_matrix)
        exp_symbol = exp(input_symbol)
        softmax_value = reduce_sum(class_symbol * exp_symbol, axis=1) / reduce_sum(exp_symbol, axis=1)
        loss = reduce_mean(-log(softmax_value))
        return loss


class SVMLoss(LossLayer):
    loss_type = LossCategory.classification

    @staticmethod
    def loss_function(input_symbol: Symbol, classification):
        class_matrix = generate_class_matrix(classification)
        dimension = class_matrix.shape[0]
        class_matrix *= -(dimension - 1)
        class_matrix[class_matrix == 0] = 1
        class_symbol = Symbol(class_matrix)
        loss = reduce_mean(maximum(reduce_sum(class_symbol * input_symbol, axis=1) + (dimension - 1), 1))
        return loss


softmax_loss = SoftMaxLoss.loss_function
svm_loss = SVMLoss.loss_function


loss_map = {
    'softmax': SoftMaxLoss,
    'svm': SVMLoss,
}


def register_loss(name: str, loss: LossLayer):
    loss_map[name.lower()] = loss


class Loss:
    def __init__(self, name: str, *args, **kwargs):
        self.__name = name.lower()
        self.__loss = None
        if self.__name in loss_map:
            self.__loss = loss_map[self.__name](*args, **kwargs)
        else:
            raise ValueError('No such loss: {}'.format(name))

    def loss_layer(self):
        return self.__loss
