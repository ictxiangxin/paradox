from paradox.symbol import *


class Engine:
    def __init__(self, symbol: Symbol=None, variable=None):
        self.__symbol = None
        self.__variables = set()
        self.__gradients = {}
        self.__shape = {}
        self.__broadcast = {}
        self.__bind = {}
        self.symbol(symbol)
        self.set_variables(variable)

    def clear(self):
        self.__gradients = {}
        self.__shape = {}
        self.__broadcast = {}

    def get_variables(self):
        return self.__variables

    def set_variables(self, symbol):
        if symbol is not None:
            old_variables = set(self.__variables)
            if isinstance(symbol, Symbol):
                symbols = {symbol}
            else:
                symbols = set(symbol)
            for symbol in symbols:
                if isinstance(symbol, Symbol):
                    self.__variables.add(symbol)
                else:
                    raise ValueError('Variable must be Symbol.')
            unused_variables = old_variables - self.__variables
            for variable in unused_variables:
                if variable in self.__gradients:
                    del self.__gradients[variable]

    variables = property(get_variables, set_variables)

    def symbol(self, symbol: Symbol):
        self.__symbol = symbol
        self.clear()
        return self

    def bind(self, bind_data: dict):
        for symbol in bind_data:
            if symbol.category == SymbolCategory.constant:
                raise ValueError('Can not bind data for Constant.')
        self.__bind = bind_data
        self.clear()
        return self

    def __compute_value(self, symbol: Symbol):
        if symbol.operator is None:
            if symbol.value is None:
                if symbol in self.__bind:
                    return numpy.array(self.__bind[symbol])
                else:
                    raise ValueError('Symbol must bind data: {}'.format(symbol))
            else:
                return symbol.value
        else:
            compute_inputs = [self.__compute_value(_s) for _s in symbol.input]
            return symbol.operator.compute(*compute_inputs)

    def __compute_gradient(self, variable: Symbol):
        if hash(self.__symbol) == hash(variable):
            self.__gradients[variable] = Symbol(1)
            return
        for forward in variable.output:
            if self.gradient(forward) is not None:
                gradients = forward.operator.gradient(*forward.input)
                index = forward.input.index(variable)
                gradient = gradients[index]
                if forward.operator.matrix and not gradient.is_scala() and not self.gradient(forward).is_scala():
                    if index == 0:
                        multiply_tuple = (self.gradient(forward), gradient)
                    else:
                        multiply_tuple = (gradient, self.gradient(forward))
                    current_gradient = multiply_tuple[0] @ multiply_tuple[1]
                else:
                    current_gradient = self.gradient(forward) * gradient
                current_broadcast = self.broadcast(variable, forward)
                for i, axis in enumerate(current_broadcast):
                    current_gradient = reduce_sum(current_gradient, axis=axis - i)
                if variable not in self.__gradients:
                    self.__gradients[variable] = current_gradient
                else:
                    self.__gradients[variable] += current_gradient

    def __compute_shape(self, symbol: Symbol):
        if symbol.operator is None:
            if symbol.value is None:
                raise ValueError('Symbol must bind data: {}'.format(symbol))
            else:
                self.__shape[symbol] = symbol.value.shape
        else:
            shape_broadcasts = symbol.operator.shape(*[self.shape(s) for s in symbol.input])
            shape = shape_broadcasts[0]
            broadcasts = shape_broadcasts[1:]
            self.__shape[symbol] = shape
            for input_symbol, input_broadcast in zip(symbol.input, broadcasts):
                if len(input_broadcast) > 0:
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
