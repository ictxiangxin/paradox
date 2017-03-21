from enum import Enum
from abc import abstractmethod
from paradox.kernel import *
from paradox.utils import generate_class_matrix


LossCategory = Enum('LossCategory', ('classification', 'regression'))


class LossFunction:
    loss_type = None

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        pass


class SoftMaxLoss(LossFunction):
    loss_type = LossCategory.classification

    @staticmethod
    def loss_function(input_symbol: Symbol, classification):
        class_matrix = generate_class_matrix(classification)
        class_symbol = Symbol(class_matrix)
        exp_symbol = exp(input_symbol)
        softmax_value = reduce_sum(class_symbol * exp_symbol, axis=0) / reduce_sum(exp_symbol, axis=0)
        loss = reduce_mean(-log(softmax_value))
        return loss


class SVMLoss(LossFunction):
    loss_type = LossCategory.classification

    @staticmethod
    def loss_function(input_symbol: Symbol, classification):
        class_matrix = generate_class_matrix(classification)
        dimension = class_matrix.shape[0]
        class_matrix *= -(dimension - 1)
        class_matrix[class_matrix == 0] = 1
        class_symbol = Symbol(class_matrix)
        loss = reduce_mean(maximum(reduce_sum(class_symbol * input_symbol, axis=0) + (dimension - 1), 0))
        return loss


softmax_loss = SoftMaxLoss.loss_function
svm_loss = SVMLoss.loss_function


loss_map = {
    'softmax': SoftMaxLoss,
    'svm': SVMLoss,
}


def register_loss(name: str, loss: LossFunction):
    loss_map[name.lower()] = loss


class Loss:
    def __init__(self, name: str):
        self.__name = name.lower()
        self.__loss = None
        if self.__name in loss_map:
            self.__loss = loss_map[self.__name]
        else:
            raise ValueError('No such loss: {}'.format(name))

    def loss_function(self):
        return self.__loss.loss_function
