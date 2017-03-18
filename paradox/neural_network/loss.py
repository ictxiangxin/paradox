from paradox.kernel import *
from paradox.utils import generate_class_matrix


def softmax_loss(input_symbol: Symbol, classification):
    class_matrix = generate_class_matrix(classification)
    class_symbol = Symbol(class_matrix)
    exp_symbol = exp(input_symbol)
    softmax_value = reduce_sum(class_symbol * exp_symbol, axis=0) / reduce_sum(exp_symbol, axis=0)
    loss = reduce_mean(-log(softmax_value))
    return loss


def svm_loss(input_symbol: Symbol, classification):
    class_matrix = generate_class_matrix(classification)
    dimension = class_matrix.shape[0]
    class_matrix *= -(dimension - 1)
    class_matrix[class_matrix == 0] = 1
    class_symbol = Symbol(class_matrix)
    loss = reduce_mean(maximum(reduce_sum(class_symbol * input_symbol, axis=0) + (dimension - 1), 0))
    return loss
