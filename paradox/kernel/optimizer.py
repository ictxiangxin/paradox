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

    def minimize(self, engine: Engine):
        engine.differentiate()
        variables = engine.variables
        for variable in variables:
            variable.value -= self.__rate * Engine(engine.gradient(variable)).value()
            engine.modified()

    def maximize(self, engine: Engine):
        engine.differentiate()
        variables = engine.variables
        for variable in variables:
            variable.value += self.__rate * Engine(engine.gradient(variable)).value()
            engine.modified()
