from paradox.kernel.operator import *
from paradox.kernel.algebra import Simplification


class Engine:
    def __init__(self, symbol: Symbol=None, variables=None):
        self.__symbol = None
        self.__variables = set()
        self.__gradients = {}
        self.__shape = {}
        self.__broadcast = {}
        self.__bind = {}
        self.__value_cache = {}
        self.__algebra_simplification = Simplification()
        self.symbol = symbol
        self.set_variables(variables)

    def clear(self):
        self.__gradients = {}
        self.__shape = {}
        self.__broadcast = {}
        self.__value_cache = {}

    def get_symbol(self):
        return self.__symbol

    def set_symbol(self, symbol: Symbol):
        self.__symbol = symbol
        self.clear()

    symbol = property(get_symbol, set_symbol)

    def get_variables(self):
        return self.__variables

    def set_variables(self, symbol):
        if symbol is None:
            symbol = set()
            symbol_set = set() if self.__symbol is None else {self.__symbol}
            while len(symbol_set):
                any_symbol = symbol_set.pop()
                if any_symbol.is_variable():
                    symbol.add(any_symbol)
                elif any_symbol.is_operator():
                    symbol_set |= set(any_symbol.input)
        old_variables = set(self.__variables)
        if isinstance(symbol, Symbol):
            symbols = {symbol}
        else:
            symbols = set(symbol)
        for symbol in symbols:
            if isinstance(symbol, Symbol) and not symbol.is_operator():
                self.__variables.add(symbol)
            else:
                raise ValueError('Variable must be Symbol.')
        unused_variables = old_variables - self.__variables
        for variable in unused_variables:
            if variable in self.__gradients:
                del self.__gradients[variable]

    variables = property(get_variables, set_variables)

    def get_bind(self):
        return self.__bind

    def set_bind(self, bind_data: dict):
        old_bind = self.__bind
        self.__bind = {}
        need_clear = False
        for s, d in bind_data.items():
            if s.category == SymbolCategory.constant:
                raise ValueError('Can not bind data for Constant.')
            d_array = numpy.array(d)
            if s in old_bind:
                if old_bind[s].shape != d_array.shape:
                    need_clear = True
            else:
                need_clear = True
            self.__bind[s] = d_array
        if need_clear:
            self.clear()

    bind = property(get_bind, set_bind)

    def get_value_cache(self):
        return self.__value_cache

    def set_value_cache(self, value_cache: dict):
        self.__value_cache = value_cache

    value_cache = property(get_value_cache, set_value_cache)

    def modified(self):
        self.__value_cache = {}

    def __compute_value(self, symbol: Symbol):
        if not symbol.is_operator():
            if symbol in self.__bind:
                return numpy.array(self.__bind[symbol])
            else:
                if symbol.value is None or symbol.is_placeholder():
                    raise ValueError('Symbol must bind data: {}'.format(symbol))
                else:
                    return symbol.value
        else:
            if symbol in self.__value_cache:
                return self.__value_cache[symbol]
            else:
                compute_inputs = [self.__compute_value(_s) for _s in symbol.input]
                symbol_value = symbol.operator.compute(*compute_inputs)
                self.__value_cache[symbol] = symbol_value
                return symbol_value

    def __compute_gradient(self, variable: Symbol):
        if hash(self.__symbol) == hash(variable):
            self.__gradients[variable] = broadcast(Constant(1), self.shape(self.__symbol))
            return
        current_operator = None
        index = -1
        for forward in variable.output:
            if self.gradient(forward) is not None:
                if current_operator != forward.operator:
                    current_operator = forward.operator
                    index = -1
                gradients = forward.operator.gradient(self, forward, *forward.input)
                for i, _variable in enumerate(forward.input, start=index + 1):
                    if hash(_variable) == hash(variable):
                        index = i
                        break
                current_gradient = gradients[index]()
                if forward.operator.auto_reduce:
                    invariant = 0
                    for i, d in enumerate(self.broadcast(variable, forward)):
                        if d > 0:
                            current_gradient = reduce_sum(current_gradient, axis=i + invariant, invariant=True)
                        elif d < 0:
                            current_gradient = reduce_sum(current_gradient, axis=i + invariant, invariant=False)
                            invariant -= 1
                if variable not in self.__gradients:
                    self.__gradients[variable] = current_gradient
                else:
                    self.__gradients[variable] += current_gradient
        if variable in self.__gradients:
            self.__algebra_simplification.simplify(self.__gradients[variable])

    def __compute_shape(self, symbol: Symbol):
        if not symbol.is_operator():
            if symbol in self.__bind:
                self.__shape[symbol] = self.__bind[symbol].shape
            else:
                if symbol.shape is None:
                    raise ValueError('Placeholder must bind data or set shape: {}'.format(symbol))
                else:
                    self.__shape[symbol] = symbol.shape
        else:
            shape_broadcasts = symbol.operator.shape(*[self.shape(s) for s in symbol.input])
            shape = shape_broadcasts[0]
            broadcasts = shape_broadcasts[1:]
            self.__shape[symbol] = shape
            for input_symbol, input_broadcast in zip(symbol.input, broadcasts):
                if sum([abs(d) for d in input_broadcast]) > 0:
                    self.__broadcast.setdefault(input_symbol, {})
                    self.__broadcast[input_symbol].setdefault(symbol, {})
                    self.__broadcast[input_symbol][symbol] = input_broadcast

    def value(self):
        return self.__compute_value(self.__symbol)

    def differentiate(self):
        for variable in self.__variables:
            if variable not in self.__gradients:
                self.__compute_gradient(variable)

    def gradient(self, variable: Symbol):
        if variable not in self.__gradients:
            self.__compute_gradient(variable)
        return self.__gradients.get(variable, None)

    def shape(self, variable: Symbol):
        if variable not in self.__shape:
            self.__compute_shape(variable)
        return self.__shape.get(variable, None)

    def broadcast(self, from_variable: Symbol, to_variable: Symbol):
        if from_variable not in self.__broadcast:
            self.__compute_shape(from_variable)
        if from_variable not in self.__broadcast:
            return ()
        else:
            if to_variable not in self.__broadcast[from_variable]:
                return ()
            else:
                return self.__broadcast[from_variable][to_variable]
