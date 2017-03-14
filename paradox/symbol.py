from enum import Enum
import numpy

SymbolCategory = Enum('SymbolCategory', ('variable', 'constant'))


class Symbol:
    def __init__(self, value=None, name: str=None, operator=None, inputs=None, category: SymbolCategory=None):
        self.__name = None
        self.__input = []
        self.__operator = None
        self.__output = []
        self.__value = None
        self.__scala = False
        self.__category = None
        self.init(value, name, operator, inputs, category)

    def init(self, value=None, name: str=None, operator=None, inputs=None, category: SymbolCategory=None):
        if isinstance(value, Symbol):
            self.name = value.name
            self.value = value.value
            for _input in value.input:
                self.__add_input(_input.clone())
            self.__set_operator(value.__operator)
            for _output in value.output:
                self.__add_output(_output.clone())
            self.__set_category(value.category)
        else:
            self.__set_value(value)
            self.arithmetic_compute(operator, inputs)
            self.__set_category(category)
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
                if self.is_scala():
                    self.__name = str(self.__value)
                else:
                    self.__name = self.__class__.__name__
        else:
            self.__name = name

    name = property(__get_name, __set_name)

    def __get_category(self):
        return self.__category

    def __set_category(self, category: SymbolCategory):
        if category is None:
            if self.__category is None:
                self.__category = SymbolCategory.variable
        else:
            if category == SymbolCategory.constant and self.__value is None:
                raise ValueError('Constant Symbol must have value.')
            else:
                self.__category = category

    category = property(__get_category, __set_category)

    def __get_value(self):
        return self.__value

    def __set_value(self, tensor):
        if tensor is not None:
            if self.__category == SymbolCategory.constant:
                raise ValueError('Can not change value for Constant')
            else:
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

    def __remove_input(self, symbol):
        new_input = []
        for o in self.__input:
            if hash(o) != hash(symbol):
                new_input.append(o)
        self.__input = new_input

    def __get_output(self):
        return self.__output

    output = property(__get_output)

    def __add_output(self, symbol):
        if isinstance(symbol, Symbol):
            self.__output.append(symbol)
        else:
            raise ValueError('Output must be Symbol class.')

    def __remove_output(self, symbol):
        new_output = []
        for o in self.__output:
            if hash(o) != hash(symbol):
                new_output.append(o)
        self.__output = new_output

    def arithmetic_compute(self, operator, inputs):
        if operator is not None and inputs:
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
        clone_symbol.__set_category(self.__category)
        return clone_symbol

    def clear_input(self):
        for symbol in set(self.__input):
            symbol.__remove_output(self)
        self.__input = []

    def clear_output(self):
        for symbol in set(self.__output):
            symbol.__remove_input(self)
        self.__output = []

    def clear_operator(self):
        self.clear_input()
        self.__operator = None

    def destroy(self):
        self.clear_input()
        self.clear_output()
        self.__output = []
        self.__value = None
        self.__operator = None

    def rebuild_name(self):
        self.__name = None
        self.__set_name(None)

    def is_scala(self):
        return self.__scala

    def is_constant(self):
        return self.__category == SymbolCategory.constant

    def is_variable(self):
        return self.__category == SymbolCategory.variable

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

    def __rpow__(self, other):
        return power(other, self)

    def __eq__(self, other):
        return equal(self, other)

    def __lt__(self, other):
        return less(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __gt__(self, other):
        return greater(self, other)

    def __ge__(self, other):
        return greater_equal(self, other)


class Constant(Symbol):
    def __init__(self, value=None, name: str=None, operator=None, inputs=None):
        Symbol.__init__(self)
        self.init(value, name, operator, inputs, SymbolCategory.constant)


class Variable(Symbol):
    def __init__(self, value=None, name: str=None, operator=None, inputs=None):
        Symbol.__init__(self)
        self.init(value, name, operator, inputs, SymbolCategory.variable)


def __as_symbol(thing):
    if isinstance(thing, Symbol):
        return thing
    else:
        return Constant(thing)


def __as_symbols(things):
    return list(map(__as_symbol, things))


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


def broadcast(a, shape):
    from paradox.operator import Broadcast
    return Symbol(operator=Broadcast(shape), inputs=__as_symbols([a]))


def reduce_sum(a, axis=None, invariant=False):
    from paradox.operator import ReduceSum
    return Symbol(operator=ReduceSum(axis, invariant), inputs=__as_symbols([a]))


def where(condition, a, b):
    from paradox.operator import Where
    return Symbol(operator=Where(), inputs=__as_symbols([condition, a, b]))


def equal(a, b):
    from paradox.operator import Equal
    return Symbol(operator=Equal(), inputs=__as_symbols([a, b]))


def not_equal(a, b):
    from paradox.operator import NotEqual
    return Symbol(operator=NotEqual(), inputs=__as_symbols([a, b]))


def less(a, b):
    from paradox.operator import Less
    return Symbol(operator=Less(), inputs=__as_symbols([a, b]))


def less_equal(a, b):
    from paradox.operator import LessEqual
    return Symbol(operator=LessEqual(), inputs=__as_symbols([a, b]))


def greater(a, b):
    from paradox.operator import Greater
    return Symbol(operator=Greater(), inputs=__as_symbols([a, b]))


def greater_equal(a, b):
    from paradox.operator import GreaterEqual
    return Symbol(operator=GreaterEqual(), inputs=__as_symbols([a, b]))


def maximum(a, b):
    from paradox.operator import Maximum
    return Symbol(operator=Maximum(), inputs=__as_symbols([a, b]))


def minimum(a, b):
    from paradox.operator import Minimum
    return Symbol(operator=Minimum(), inputs=__as_symbols([a, b]))
