import numpy


class Symbol:
    def __init__(self, value=None, name: str=None, operator=None, inputs=None):
        self.__name = None
        self.__input = []
        self.__operator = None
        self.__output = []
        self.__value = None
        self.__scala = False
        if isinstance(value, Symbol):
            self.name = value.name
            self.value = value.value
            for _input in value.input:
                self.__add_input(_input.clone())
            self.__set_operator(value.__operator)
            for _output in value.output:
                self.__add_output(_output.clone())
        else:
            self.__set_value(value)
            self.__create_compute(operator, inputs)
            self.__set_name(name)

    def __repr__(self):
        if self.__operator is None:
            return self.__name
        else:
            if self.__operator.operator_sign is None:
                arguments = list(map(str, self.input))
                arguments += ['{}={}'.format(k, v) for k, v in self.__operator.arguments.items()]
                return '{}({})'.format(self.__operator.__class__.__name__, ', '.join(arguments))
            else:
                return '({} {} {})'.format(self.input[0], self.__operator.operator_sign, self.input[1])

    def __str__(self):
        return self.__repr__()

    def __get_name(self):
        return self.__name

    def __set_name(self, name: str):
        if name is None:
            if self.__value is None:
                self.__name = self.__class__.__name__
            else:
                self.__name = str(self.__value)
        else:
            self.__name = name

    name = property(__get_name, __set_name)

    def __get_value(self):
        return self.__value

    def __set_value(self, tensor):
        if tensor is not None:
            self.__value = numpy.array(tensor, dtype=float)
            self.__scala = len(self.value.shape) == 0

    value = property(__get_value, __set_value)

    def __get_operator(self):
        return self.__operator

    def __set_operator(self, operator):
        from paradox.operator import Operator
        if operator is not None:
            if isinstance(operator, Operator):
                self.__operator = operator
            else:
                raise ValueError('Operator must be Operator class.')

    operator = property(__get_operator, __set_operator)

    def __get_input(self):
        return self.__input

    input = property(__get_input)

    def __add_input(self, symbol):
        if isinstance(symbol, Symbol):
            self.__input.append(symbol)
        else:
            raise ValueError('Input must be Symbol class.')

    def __get_output(self):
        return self.__output

    output = property(__get_output)

    def __add_output(self, symbol):
        if isinstance(symbol, Symbol):
            self.__output.append(symbol)
        else:
            raise ValueError('Input must be Symbol class.')

    def __create_compute(self, operator, inputs):
        if operator is not None or inputs is not None:
            self.__set_operator(operator)
            inputs_count = operator.inputs_count
            if inputs_count is None:
                inputs_count = len(inputs)
            self.__input = []
            self.__scala = True
            for symbol in inputs[:inputs_count]:
                if isinstance(symbol, Symbol):
                    if not symbol.is_scala():
                        self.__scala = False
                    self.__add_input(symbol)
                    symbol.__add_output(self)
                else:
                    raise ValueError('Input must be Symbol class.')

    def clone(self):
        clone_symbol = Symbol()
        clone_symbol.name = self.name
        clone_symbol.value = self.value
        for _input in self.input:
            clone_symbol.__add_input(_input.clone())
        clone_symbol.__set_operator(self.__operator)
        for _output in self.output:
            clone_symbol.__add_output(_output.clone())
        return clone_symbol

    def is_scala(self):
        return self.__scala

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        return plus(self, other)

    def __radd__(self, other):
        return plus(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __matmul__(self, other):
        return matrix_multiply(self, other)

    def __rmatmul__(self, other):
        return matrix_multiply(other, self)

    def __pow__(self, exponent):
        return power(self, exponent)


def __as_symbol(obj):
    if isinstance(obj, Symbol):
        return obj
    else:
        return Symbol(obj)


def __as_symbols(objs):
    return list(map(__as_symbol, objs))


def plus(a, b):
    from paradox.operator import Plus
    return Symbol(operator=Plus(), inputs=__as_symbols([a, b]))


def subtract(a, b):
    from paradox.operator import Subtract
    return Symbol(operator=Subtract(), inputs=__as_symbols([a, b]))


def multiply(a, b):
    from paradox.operator import Multiply
    return Symbol(operator=Multiply(), inputs=__as_symbols([a, b]))


def divide(a, b):
    from paradox.operator import Divide
    return Symbol(operator=Divide(), inputs=__as_symbols([a, b]))


def matrix_multiply(a, b):
    from paradox.operator import MatrixMultiply
    return Symbol(operator=MatrixMultiply(), inputs=__as_symbols([a, b]))


def power(a, b):
    from paradox.operator import Power
    return Symbol(operator=Power(), inputs=__as_symbols([a, b]))


def log(a):
    from paradox.operator import Log
    return Symbol(operator=Log(), inputs=__as_symbols([a]))


def transpose(a, axes=None):
    from paradox.operator import Transpose
    return Symbol(operator=Transpose(axes), inputs=__as_symbols([a]))


def reduce_sum(a, axis=None, invariant=False):
    from paradox.operator import ReduceSum
    return Symbol(operator=ReduceSum(axis, invariant), inputs=__as_symbols([a]))
