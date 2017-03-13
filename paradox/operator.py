from abc import abstractmethod
from paradox.symbol import *


def element_wise_shape(shape_a, shape_b):
    gap = abs(len(shape_a) - len(shape_b))
    if len(shape_a) > len(shape_b):
        new_shape = list(shape_a)
        broadcast_a = []
        broadcast_b = list(range(gap))
    else:
        new_shape = list(shape_b)
        broadcast_a = list(range(gap))
        broadcast_b = []
    for i in range(min(len(shape_a), len(shape_b))):
        if shape_a[i] == 1 or shape_b[i] == 1:
            if shape_a[i] > shape_b[i]:
                new_shape[i] = shape_a[i]
                broadcast_b.append(i + gap)
            elif shape_b[i] > shape_a[i]:
                new_shape[i] = shape_b[i]
                broadcast_a.append(i + gap)
        elif shape_a[i] != shape_b[i]:
            raise ValueError('Can not broadcast these two shapes: a={}, b={}'.format(shape_a, shape_b))
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
            gap = abs(len(shape_a) - len(shape_b))
            if len(shape_a) > len(shape_b):
                if shape_a[gap:-2] != shape_b[:-2]:
                    raise ValueError()
                new_shape = list(shape_a)
                broadcast_a = ()
                broadcast_b = tuple(range(gap))
            else:
                if shape_b[gap:-2] != shape_a[:-2]:
                    raise ValueError()
                new_shape = list(shape_b)
                broadcast_a = tuple(range(gap))
                broadcast_b = ()
            new_shape[-1] = shape_b[-1]
            new_shape[-2] = shape_a[-2]
            return tuple(new_shape), broadcast_a, broadcast_b
        else:
            raise ValueError()
    except ValueError:
        raise ValueError('Can not execute matrix multiply these two shapes: a={}, b={}'.format(shape_a, shape_b))


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


def transpose_shape(shape_a, axes):
    if axes is None:
        return tuple(reversed(shape_a)), (), ()
    else:
        if len(set(axes)) == len(axes):
            if set(axes) == set(range(len(axes))) and len(axes) == len(shape_a):
                new_shape = [0] * len(shape_a)
                for i, d in zip(axes, shape_a):
                    new_shape[i] = d
                return tuple(new_shape), (), ()
            else:
                ValueError('Invalid axes for this Shape: shape={}, axes={}'.format(shape_a, axes))
        else:
            ValueError('Repeated axis in axes: {}'.format(axes))


class Operator:
    operator_sign = None
    inputs_count = None
    value_dependent = False
    arguments = {}

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

    def compute(self, value_a, value_b):
        return value_a + value_b

    def gradient(self, forward, symbol_a, symbol_b):
        return [lambda: forward * Constant(1),
                lambda: forward * Constant(1)]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Subtract(Operator):
    def __init__(self):
        self.operator_sign = '-'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return value_a - value_b

    def gradient(self, forward, symbol_a, symbol_b):
        return [lambda: forward * Constant(1),
                lambda: forward * Constant(-1)]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Multiply(Operator):
    def __init__(self):
        self.operator_sign = '*'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return value_a * value_b

    def gradient(self, forward, symbol_a, symbol_b):
        return [lambda: forward * symbol_b,
                lambda: forward * symbol_a]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Divide(Operator):
    def __init__(self):
        self.operator_sign = '/'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return value_a / value_b

    def gradient(self, forward, symbol_a, symbol_b):
        return [lambda: forward * Constant(1) / symbol_b,
                lambda: forward * Constant(-1) * symbol_a / (symbol_b ** Constant(2))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class MatrixMultiply(Operator):
    def __init__(self):
        self.operator_sign = '@'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        if len(value_a.shape) == 0 or len(value_b.shape) == 0:
            return value_a * value_b
        else:
            return value_a @ value_b

    def gradient(self, forward, symbol_a, symbol_b):
        return [lambda: forward @ transpose(symbol_b),
                lambda: transpose(symbol_a) @ forward]

    def shape(self, shape_a, shape_b):
        return matrix_multiply_shape(shape_a, shape_b)


class Transpose(Operator):
    def __init__(self, axes=None):
        self.inputs_count = 1
        self.arguments = {'axes': axes}

    def compute(self, value_a):
        return numpy.transpose(value_a, axes=self.arguments['axes'])

    def gradient(self, forward, symbol_a):
        return [lambda: transpose(forward, axes=self.arguments['axes'])]

    def shape(self, shape_a):
        return transpose_shape(shape_a, self.arguments['axes'])


class ReduceSum(Operator):
    def __init__(self, axis: int=None, invariant: bool=False):
        self.inputs_count = 1
        self.arguments = {'axis': axis, 'invariant': invariant}

    def compute(self, value_a):
        return numpy.sum(value_a, axis=self.arguments['axis'], keepdims=self.arguments['invariant'])

    def gradient(self, forward, symbol_a):
        return [lambda: forward]

    def shape(self, shape_a):
        return reduce_shape(shape_a, **self.arguments)


class Power(Operator):
    def __init__(self):
        self.operator_sign = '**'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.power(value_a, value_b)

    def gradient(self, forward, symbol_a, symbol_b):
        return [lambda: forward * symbol_b * (symbol_a ** (symbol_b - Constant(1))),
                lambda: forward * (symbol_a ** symbol_b) * log(symbol_a)]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Log(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.log(value_a)

    def gradient(self, forward, symbol_a):
        return [lambda: forward * Constant(1) / symbol_a]

    def shape(self, shape_a):
        return shape_a, (), ()


class BinaryCondition(Operator):
    def __init__(self, judgement):
        self.inputs_count = 2
        self.value_dependent = True
        self.judgement = judgement
        self.values = None

    def compute(self, value_a, value_b):
        self.values = [value_a, value_b]
        if self.judgement(value_a, value_b):
            return value_a
        else:
            return value_b

    def gradient(self, forward, symbol_a, symbol_b):
        if self.judgement(*self.values):
            return [lambda: forward,
                    lambda: Constant(0)]
        else:
            return [lambda: Constant(0),
                    lambda: forward]

    def shape(self, shape_a, shape_b):
        if self.judgement(*self.values):
            return shape_a
        else:
            return shape_b


class Max(BinaryCondition):
    def __init__(self):
        BinaryCondition.__init__(self, judgement=lambda a, b: a > b)
