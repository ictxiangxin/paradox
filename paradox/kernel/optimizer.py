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


class AdaptiveGradientOptimizer(Optimizer):
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


class AdaptiveDeltaOptimizer(Optimizer):
    def __init__(self, decay: float, consistent: bool=False):
        self.__decay = decay
        self.__consistent = consistent
        self.__gradient_engine = Engine()
        self.__accumulate_gradient_map = {}
        self.__expectation_map = {}

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
            self.__expectation_map.setdefault(variable, 0)
            self.__accumulate_gradient_map[variable] = self.__decay * self.__accumulate_gradient_map[variable] + (1 - self.__decay) * current_gradient ** 2
            delta = (self.__expectation_map[variable] + 1e-8) ** 0.5 / (self.__accumulate_gradient_map[variable] + 1e-8) ** 0.5 * current_gradient
            self.__expectation_map[variable] = self.__decay * self.__expectation_map[variable] + (1 - self.__decay) * delta ** 2
            variable.value = calculate_function(variable.value, delta)
        engine.modified()
        self.__gradient_engine.modified()

    def minimize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v - g)

    def maximize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v + g)


class RootMeanSquarePropOptimizer(Optimizer):
    def __init__(self, rate: float, consistent: bool=False):
        self.__rate = rate
        self.__consistent = consistent
        self.__gradient_engine = Engine()
        self.__mean_map = {}
        self.__step = 1

    def optimize(self, engine: Engine, calculate_function):
        variables = engine.variables
        for variable in variables:
            value_cache = self.__gradient_engine.value_cache
            self.__gradient_engine.symbol = engine.gradient(variable)
            self.__gradient_engine.bind = engine.bind
            if self.__consistent:
                self.__gradient_engine.value_cache = value_cache
            current_gradient = self.__gradient_engine.value()
            self.__mean_map.setdefault(variable, 0)
            self.__mean_map[variable] *= (self.__step - 1) / self.__step
            self.__mean_map[variable] += current_gradient ** 2 / self.__step
            self.__step += 1
            regularization_value = current_gradient / (self.__mean_map[variable] + 1e-8) ** 0.5
            variable.value = calculate_function(variable.value, self.__rate * regularization_value)
        engine.modified()
        self.__gradient_engine.modified()

    def minimize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v - g)

    def maximize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v + g)


class AdaptiveMomentEstimationOptimizer(Optimizer):
    def __init__(self, rate: float, decay: float=0.9, square_decay: float=0.999, consistent: bool=False):
        self.__rate = rate
        self.__decay = decay
        self.__square_decay = square_decay
        self.__consistent = consistent
        self.__gradient_engine = Engine()
        self.__estimation_map = {}
        self.__square_estimation_map = {}
        self.__step = 1

    def optimize(self, engine: Engine, calculate_function):
        variables = engine.variables
        for variable in variables:
            value_cache = self.__gradient_engine.value_cache
            self.__gradient_engine.symbol = engine.gradient(variable)
            self.__gradient_engine.bind = engine.bind
            if self.__consistent:
                self.__gradient_engine.value_cache = value_cache
            current_gradient = self.__gradient_engine.value()
            self.__estimation_map.setdefault(variable, 0)
            self.__square_estimation_map.setdefault(variable, 0)
            self.__estimation_map[variable] = self.__decay * self.__estimation_map[variable] + (1 - self.__decay) * current_gradient
            self.__square_estimation_map[variable] = self.__square_decay * self.__square_estimation_map[variable] + (1 - self.__square_decay) * current_gradient ** 2
            estimation = self.__estimation_map[variable] / (1 - self.__decay ** self.__step)
            square_estimation = self.__square_estimation_map[variable] / (1 - self.__square_decay ** self.__step)
            self.__step += 1
            regularization_value = estimation / (square_estimation + 1e-8) ** 0.5
            variable.value = calculate_function(variable.value, self.__rate * regularization_value)
        engine.modified()
        self.__gradient_engine.modified()

    def minimize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v - g)

    def maximize(self, engine: Engine):
        self.optimize(engine, lambda v, g: v + g)
