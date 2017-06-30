from abc import abstractmethod
from paradox.kernel.engine import Engine


class Optimizer:
    @abstractmethod
    def minimize(self, engine: Engine):
        pass

    @abstractmethod
    def maximize(self, engine: Engine):
        pass


class GradientDescentOptimizer(Optimizer):
    def __init__(self, rate: float, consistent: bool=False):
        self.__rate = rate
        self.__consistent = consistent
        self.__gradient_engine = Engine()

    def optimize(self, engine: Engine, calculate_function):
        variables = engine.variables
        for variable in variables:
            value_cache = self.__gradient_engine.value_cache
            self.__gradient_engine.symbol = engine.gradient(variable)
            self.__gradient_engine.bind = engine.bind
            if self.__consistent:
                self.__gradient_engine.value_cache = value_cache
            variable.value = calculate_function(variable.value, self.__rate * self.__gradient_engine.value())
        engine.modified()
        self.__gradient_engine.modified()

    def minimize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v - g)

    def maximize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v + g)


class MomentumOptimizer(Optimizer):
    def __init__(self, rate: float, factor: float,  consistent: bool=False):
        self.__rate = rate
        self.__factor = factor
        self.__consistent = consistent
        self.__old_gradient_map = {}
        self.__gradient_engine = Engine()

    def optimize(self, engine: Engine, calculate_function):
        variables = engine.variables
        for variable in variables:
            value_cache = self.__gradient_engine.value_cache
            self.__gradient_engine.symbol = engine.gradient(variable)
            self.__gradient_engine.bind = engine.bind
            if self.__consistent:
                self.__gradient_engine.value_cache = value_cache
            momentum = self.__gradient_engine.value() + self.__factor * self.__old_gradient_map.get(variable, 0)
            self.__old_gradient_map[variable] = momentum
            variable.value = calculate_function(variable.value, self.__rate * momentum)
        engine.modified()
        self.__gradient_engine.modified()

    def minimize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v - g)

    def maximize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v + g)


class AdaGradientOptimizer(Optimizer):
    def __init__(self, rate: float, consistent: bool=False):
        self.__rate = rate
        self.__consistent = consistent
        self.__gradient_engine = Engine()
        self.__accumulate_gradient_map = {}

    def optimize(self, engine: Engine, calculate_function):
        variables = engine.variables
        for variable in variables:
            value_cache = self.__gradient_engine.value_cache
            self.__gradient_engine.symbol = engine.gradient(variable)
            self.__gradient_engine.bind = engine.bind
            if self.__consistent:
                self.__gradient_engine.value_cache = value_cache
            current_gradient = self.__gradient_engine.value()
            self.__accumulate_gradient_map.setdefault(variable, 0)
            self.__accumulate_gradient_map[variable] += current_gradient ** 2
            regularization_value = current_gradient / (self.__accumulate_gradient_map[variable] + 1e-8) ** 0.5
            variable.value = calculate_function(variable.value, self.__rate * regularization_value)
        engine.modified()
        self.__gradient_engine.modified()

    def minimize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v - g)

    def maximize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v + g)
