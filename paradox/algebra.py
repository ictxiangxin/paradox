from paradox.operator import *


class Template:
    active_operator = None

    def __init__(self):
        self.active_sign = self.active_operator.__name__

    @staticmethod
    def reduce_left(symbol: Symbol):
        left_symbol, right_symbol = symbol.input
        symbol.value = left_symbol.value
        symbol.name = left_symbol.name
        symbol.clear_operator()
        symbol.arithmetic_compute(left_symbol.operator, left_symbol.input)
        right_symbol.destroy()

    @staticmethod
    def reduce_right(symbol: Symbol):
        left_symbol, right_symbol = symbol.input
        symbol.value = right_symbol.value
        symbol.name = right_symbol.name
        symbol.clear_operator()
        symbol.arithmetic_compute(right_symbol.operator, right_symbol.input)
        left_symbol.destroy()

    @abstractmethod
    def simplify(self, symbol: Symbol):
        pass


class TemplatePlus(Template):
    active_operator = Plus

    def simplify(self, symbol: Symbol):
        left_symbol, right_symbol = symbol.input
        if left_symbol.value == 0:
            self.reduce_right(symbol)
            return True
        elif right_symbol.value == 0:
            self.reduce_left(symbol)
            return True
        else:
            return False


class TemplateMultiply(Template):
    active_operator = Multiply

    def simplify(self, symbol: Symbol):
        left_symbol, right_symbol = symbol.input
        if left_symbol.value == 1:
            self.reduce_right(symbol)
            return True
        elif right_symbol.value == 1:
            self.reduce_left(symbol)
            return True
        else:
            return False


class Simplification:
    def __init__(self):
        default_templates = [
            TemplatePlus,
            TemplateMultiply,
        ]
        self.__templates = {}
        for template in default_templates:
            self.register(template())

    def operator_trigger(self, operator: Operator):
        operator_sign = operator.__class__.__name__
        if operator_sign in self.__templates:
            return self.__templates[operator_sign]
        else:
            set()

    def register(self, template: Template):
        active_operator = template.active_sign
        self.__templates.setdefault(active_operator, set())
        self.__templates[active_operator].add(template)

    def simplify(self, symbol: Symbol):
        while True:
            templates = self.operator_trigger(symbol.operator)
            simplified = True
            if templates:
                for template in templates:
                    simplified &= template.simplify(symbol)
                if not simplified:
                    break
            else:
                break
        for next_symbol in symbol.input:
            if next_symbol.operator is not None:
                self.simplify(next_symbol)
