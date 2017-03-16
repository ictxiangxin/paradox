from paradox.kernel import *
from paradox.utils import generate_class_matrix


def softmax_loss(input_symbol: Symbol, classification, name: str=None):
    class_matrix = generate_class_matrix(classification)
    class_symbol = Symbol(class_matrix, name='Classification' if name is None else name)
    exp_symbol = exp(input_symbol)
    softmax_value = reduce_sum(class_symbol * exp_symbol, axis=0) / reduce_sum(exp_symbol, axis=0)
    loss = reduce_mean(-log(softmax_value))
    return loss
