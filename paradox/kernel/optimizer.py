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
    def __init__(self, rate: float):
        self.__rate = rate
        self.__gradient_engine = Engine()

    def minimize(self, engine: Engine):
        variables = engine.variables
        for variable in variables:
            self.__gradient_engine.symbol = engine.gradient(variable)
            self.__gradient_engine.bind = engine.bind
            variable.value -= self.__rate * self.__gradient_engine.value()
            engine.modified()

    def maximize(self, engine: Engine):
        variables = engine.variables
        for variable in variables:
            self.__gradient_engine.symbol(engine.gradient(variable))
            self.__gradient_engine.bind = engine.bind
            variable.value += self.__rate * self.__gradient_engine.value()
            engine.modified()
