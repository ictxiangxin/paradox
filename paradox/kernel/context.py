from paradox.kernel.symbol import Symbol, SymbolCategory
from paradox.kernel.engine import Engine


class Context:
    def __init__(self, symbol: Symbol=None, variables=None):
        self.__engine: Engine = None
        self.__variables = []
        self.__value_cache = {}
        self.__gradient_cache = {}
        self.__gradient_engines = {}
        self.__symbol_dependence_variables = {}
        self.set_engine(Engine(symbol, variables))

    def get_engine(self):
        return self.__engine

    def set_engine(self, engine: Engine):
        if hash(self.__engine) == hash(engine):
            self.__engine = engine
            self.__variables = self.__engine.variables
            self.clear()

    engine = property(get_engine, set_engine)

    def clear(self):
        self.__value_cache = {}
        self.__gradient_cache = {}

    def initialization(self):
        for symbol in self.__variables:
            gradient_symbol = self.__engine.gradient(symbol)
            self.__gradient_engines[symbol] = Engine(gradient_symbol)
            self.__symbol_dependence_variables[symbol] = gradient_symbol.dependenc(SymbolCategory.variable)
