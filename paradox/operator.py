from abc import abstractmethod
from paradox.symbol import *


def binary_shape(shape_a, shape_b):
    if len(shape_a) > len(shape_b):
        new_shape = list(shape_a)
        base_shape = shape_b
        broadcast_a_shape = [0] * len(shape_a)
        broadcast_b_shape = list(shape_a)
    elif len(shape_a) < len(shape_b):
        new_shape = list(shape_b)
        base_shape = shape_a
        broadcast_a_shape = list(shape_b)
        broadcast_b_shape = [0] * len(shape_b)
    else:
        new_shape = list(shape_a)
        base_shape = shape_a
        broadcast_a_shape = [0] * len(shape_a)
        broadcast_b_shape = [0] * len(shape_b)
    for i in range(len(base_shape)):
        if shape_a[i] == 1 or shape_b[i] == 1:
            if shape_a[i] > shape_b[i]:
                new_shape[i] = shape_a[i]
                broadcast_b_shape[i] = shape_a[i]
            elif shape_b[i] > shape_a[i]:
                new_shape[i] = shape_b[i]
                broadcast_a_shape[i] = shape_b[i]
        elif shape_a[i] != shape_b[i]:
            raise ValueError('Can not broadcast these two shapes: {}, {}'.format(shape_a, shape_b))
    broadcast_a = []
    broadcast_b = []
    for i, (v_a, v_b) in enumerate(zip(broadcast_a_shape, broadcast_b_shape)):
        if v_a != 0:
            broadcast_a.append(i)
        if v_b != 0:
            broadcast_b.append(i)
    return tuple(new_shape), tuple(broadcast_a), tuple(broadcast_b)


def matrix_multiply_shape(shape_a, shape_b):
    try:
        if len(shape_a) == 0 or len(shape_b) == 0:
            raise ValueError()
        if len(shape_a) == 1 and len(shape_b) == 1:
            if shape_a[0] == shape_b[0]:
                return (), (), ()
            else:
                raise ValueError()
        if len(shape_a) == 1:
            if shape_a[0] == shape_b[-2]:
                new_shape = list(shape_b)
                del new_shape[-2]
                return tuple(new_shape), (), ()
            else:
                raise ValueError()
        if len(shape_b) == 1:
            if shape_a[-1] == shape_b[0]:
                return shape_a[:-1], (), ()
            else:
                raise ValueError()
        if shape_a[-1] == shape_b[-2]:
            distance = abs(len(shape_a) - len(shape_b))
            if len(shape_a) > len(shape_b):
                if shape_a[distance:-2] != shape_b[:-2]:
                    raise ValueError()
                new_shape = list(shape_a)
                broadcast_a = ()
                broadcast_b = tuple(range(distance))
            else:
                if shape_b[distance:-2] != shape_a[:-2]:
                    raise ValueError()
                new_shape = list(shape_b)
                broadcast_a = tuple(range(distance))
                broadcast_b = ()
            new_shape[-1] = shape_b[-1]
            new_shape[-2] = shape_a[-2]
            return tuple(new_shape), broadcast_a, broadcast_b
        else:
            raise ValueError()
    except ValueError:
        raise ValueError('Can not execute matrix multiply these two shapes: {}, {}'.format(shape_a, shape_b))


def reduce_shape(shape_a, axis, invariant):
    if axis is None:
        return (), (), ()
    else:
        new_shape = list(shape_a)
        if invariant:
            new_shape[axis] = 1
        else:
            del new_shape[axis]
        return tuple(new_shape), (), ()


class Operator:
    operator_sign = None
    inputs_count = None
    arguments = {}
    matrix = False

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    @abstractmethod
    def gradient(self, *args, **kwargs):
        pass

    @abstractmethod
    def shape(self, *args, **kwargs):
        pass


class Plus(Operator):
    def __init__(self):
        self.operator_sign = '+'
        self.inputs_count = 2
        self.matrix = False

    def compute(self, a, b):
        return a + b

    def gradient(self, a, b):
        return [Symbol(1), Symbol(1)]

    def shape(self, shape_a, shape_b):
        return binary_shape(shape_a, shape_b)


class Subtract(Operator):
    def __init__(self):
        self.operator_sign = '-'
        self.inputs_count = 2
        self.matrix = False

    def compute(self, a, b):
        return a - b

    def gradient(self, a, b):
        return [Symbol(1), Symbol(-1)]

    def shape(self, shape_a, shape_b):
        return binary_shape(shape_a, shape_b)


class Multiply(Operator):
    def __init__(self):
        self.operator_sign = '*'
        self.inputs_count = 2
        self.matrix = False

    def compute(self, a, b):
        return a * b

    def gradient(self, a, b):
        return [b, a]

    def shape(self, shape_a, shape_b):
        return binary_shape(shape_a, shape_b)


class Divide(Operator):
    def __init__(self):
        self.operator_sign = '/'
        self.inputs_count = 2
        self.matrix = False

    def compute(self, a, b):
        return a / b

    def gradient(self, a, b):
        return [Symbol(1) / b, Symbol(-1) * a / (b * b)]

    def shape(self, shape_a, shape_b):
        return binary_shape(shape_a, shape_b)


class MatrixMultiply(Operator):
    def __init__(self):
        self.operator_sign = '@'
        self.inputs_count = 2
        self.matrix = True

    def compute(self, a, b):
        return a @ b

    def gradient(self, a, b):
        return [transpose(b), transpose(a)]

    def shape(self, shape_a, shape_b):
        return matrix_multiply_shape(shape_a, shape_b)


class Transpose(Operator):
    def __init__(self):
        self.inputs_count = 1
        self.matrix = False

    def compute(self, a):
        return numpy.transpose(a)

    def gradient(self, a):
        return [Symbol(1)]

    def shape(self, shape_a):
        return tuple(reversed(shape_a)), (), ()


class ReduceSum(Operator):
    def __init__(self, axis: int=None, invariant: bool=False):
        self.inputs_count = 1
        self.arguments = {'axis': axis, 'invariant': invariant}
        self.matrix = False

    def compute(self, a):
        return numpy.sum(a, axis=self.arguments['axis'], keepdims=self.arguments['invariant'])

    def gradient(self, a):
        return [Symbol(1)]

    def shape(self, shape_a):
        return reduce_shape(shape_a, **self.arguments)


class Power(Operator):
    def __init__(self):
        self.operator_sign = '**'
        self.inputs_count = 2
        self.matrix = False

    def compute(self, a, b):
        return numpy.power(a, b)

    def gradient(self, a, b):
        return [b * (a ** (b - 1)), (a ** b) * log(a)]

    def shape(self, shape_a, shape_b):
        return binary_shape(shape_a, shape_b)


class Log(Operator):
    def __init__(self):
        self.inputs_count = 1
        self.matrix = False

    def compute(self, a):
        return numpy.log(a)

    def gradient(self, a):
        return [1 / a]

    def shape(self, shape_a):
        return shape_a, (), ()
