from paradox.kernel.operator import *


def equal(a, b):
    result = a == b
    if isinstance(result, bool):
        return result
    else:
        return result.all()


class Template:
    active_operator = None

    def __init__(self):
        if self.active_operator is None:
            self.active_sign = None
        else:
            self.active_sign = self.active_operator.__name__

    @staticmethod
    def reduce_symbol(symbol: Symbol, index: int):
        input_list = symbol.input
        symbol.value = input_list[index].value
        symbol.name = input_list[index].name
        symbol.clear_operator()
        symbol.symbolic_compute(input_list[index].operator, input_list[index].input)
        for i in range(len(input_list)):
            if i != index:
                input_list[i].destroy()

    @abstractmethod
    def simplify(self, symbol: Symbol):
        pass


class TemplateConstant(Template):
    active_operator = None

    def simplify(self, symbol: Symbol):
        if symbol.is_operator():
            for s in symbol.input:
                if not s.is_constant():
                    return False
            compute_inputs = [_s.value for _s in symbol.input]
            value = symbol.operator.compute(*compute_inputs)
            symbol.clear_operator()
            symbol.value = value
            symbol.category = SymbolCategory.constant
            symbol.rebuild_name()
            return True
        else:
            return False


class TemplatePlus(Template):
    active_operator = Plus

    def simplify(self, symbol: Symbol):
        left_symbol, right_symbol = symbol.input
        if equal(left_symbol.value, 0) and left_symbol.is_constant():
            self.reduce_symbol(symbol, 1)
            return True
        elif equal(right_symbol.value, 0) and right_symbol.is_constant():
            self.reduce_symbol(symbol, 0)
            return True
        else:
            return False


class TemplateSubtract(Template):
    active_operator = Plus

    def simplify(self, symbol: Symbol):
        left_symbol, right_symbol = symbol.input
        if equal(left_symbol.value, 0) and left_symbol.is_constant():
            symbol.clear_operator()
            symbol.clear_input()
            symbol.symbolic_compute(Negative(), [right_symbol])
            return True
        elif equal(right_symbol.value, 0) and right_symbol.is_constant():
            self.reduce_symbol(symbol, 0)
            return True
        else:
            return False


class TemplateMultiply(Template):
    active_operator = Multiply

    def simplify(self, symbol: Symbol):
        left_symbol, right_symbol = symbol.input
        if equal(left_symbol.value, 1) and left_symbol.is_constant():
            self.reduce_symbol(symbol, 1)
            return True
        elif equal(right_symbol.value, 1) and right_symbol.is_constant():
            self.reduce_symbol(symbol, 0)
            return True
        else:
            return False


default_templates = [
    TemplateConstant,
    TemplatePlus,
    TemplateSubtract,
    TemplateMultiply,
]


class Simplification:
    def __init__(self):
        self.__templates = {}
        for template in default_templates:
            self.register(template())

    def operator_trigger(self, operator: Operator):
        operator_sign = operator.__class__.__name__
        if operator_sign in self.__templates:
            return self.__templates[operator_sign]
        else:
            return set()

    def register(self, template: Template):
        active_operator = template.active_sign
        self.__templates.setdefault(active_operator, set())
        self.__templates[active_operator].add(template)

    def simplify(self, symbol: Symbol):
        while self.simplify_cycle(symbol):
            pass

    def simplify_cycle(self, symbol: Symbol):
        effective = False
        templates = self.operator_trigger(symbol.operator) | self.__templates[None]
        if templates:
            for template in templates:
                if template.simplify(symbol):
                    effective |= True
                    break
        for next_symbol in symbol.input:
            if next_symbol.operator is not None:
                effective |= self.simplify_cycle(next_symbol)
        return effective

