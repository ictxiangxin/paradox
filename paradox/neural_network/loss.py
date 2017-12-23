from enum import Enum
from abc import abstractmethod
from paradox.kernel import *
from paradox.utils import generate_label_matrix


class LossCategory(Enum):
    classification = 0
    regression = 1


class LossLayer:
    loss_type = None

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        pass


class SoftMaxLoss(LossLayer):
    loss_type = LossCategory.classification

    @staticmethod
    def loss_function(input_symbol: Symbol, label_symbol: Symbol):
        exp_symbol = exp(input_symbol)
        softmax_value = reduce_sum(label_symbol * exp_symbol, axis=1) / reduce_sum(exp_symbol, axis=1)
        loss = reduce_mean(-log(softmax_value))
        return loss


class SVMLoss(LossLayer):
    loss_type = LossCategory.classification

    @staticmethod
    def loss_function(input_symbol: Symbol, label_symbol: Symbol):
        dimension = label_symbol.shape[1]
        label_symbol = label_symbol * -(dimension - 1)
        label_symbol = where(label_symbol == 0, 1, label_symbol)
        loss = reduce_mean(maximum(reduce_sum(label_symbol * input_symbol, axis=1) + (dimension - 1), 0))
        return loss


softmax_loss = SoftMaxLoss.loss_function
svm_loss = SVMLoss.loss_function


def softmax_loss_with_label(input_symbol: Symbol, classification):
    label_symbol = Constant(generate_label_matrix(classification)[0])
    return softmax_loss(input_symbol, label_symbol)


def svm_loss_with_label(input_symbol: Symbol, classification):
    label_symbol = Constant(generate_label_matrix(classification)[0])
    return svm_loss(input_symbol, label_symbol)


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
