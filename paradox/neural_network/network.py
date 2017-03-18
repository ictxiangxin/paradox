import numpy
from paradox.kernel.symbol import Variable
from paradox.kernel.engine import Engine
from paradox.kernel.optimizer import GradientDescentOptimizer
from paradox.neural_network.connection import Connection
from paradox.neural_network.activation import Activation
from paradox.neural_network.loss import softmax_loss, svm_loss

optimizer_map = {
    'gd': GradientDescentOptimizer,
}

loss_map = {
    'softmax': softmax_loss,
    'svm': svm_loss,
}


class Network:
    def __init__(self):
        self.__layer = []
        self.__input_symbol = Variable(name='InputSymbol')
        self.__current_symbol = self.__input_symbol
        self.__current_output = None
        self.__variables = []
        self.__data = None
        self.__optimizer = None
        self.__loss = None
        self.__train_engine = Engine()
        self.__predict_engine = Engine()

    def add(self, layer):
        if isinstance(layer, Connection):
            weight, bias = layer.weight_bias(self.__current_output)
            self.__variables.append(weight)
            self.__variables.append(bias)
            self.__current_symbol = weight @ self.__current_symbol + bias
            self.__current_output = layer.output_dimension()
        elif isinstance(layer, Activation):
            activation_function = layer.activation_function()
            self.__current_symbol = activation_function(self.__current_symbol)
        else:
            raise ValueError('Invalid layer type: {}'. format(type(layer)))

    def optimizer(self, name: str, *args, **kwargs):
        name = name.lower()
        if name in optimizer_map:
            self.__optimizer = optimizer_map[name](*args, **kwargs)
        else:
            raise ValueError('No such optimizer: {}'.format(name))

    def loss(self, name: str):
        name = name.lower()
        if name in loss_map:
            self.__loss = loss_map[name]
        else:
            raise ValueError('No such loss: {}'.format(name))

    def train(self, data, target, epochs: int=10000, loss_threshold: float=0.001, state_cycle: int=100):
        loss = self.__loss(self.__current_symbol, target)
        self.__train_engine.symbol = loss
        self.__train_engine.variables = self.__variables
        self.__train_engine.bind = {self.__input_symbol: data}
        for variable in self.__variables:
            variable.value = numpy.random.normal(0, 1, variable.value.shape)
        self.__loss_value = []
        for epoch in range(epochs):
            self.__optimizer.minimize(self.__train_engine)
            if (epoch + 1) % state_cycle == 0:
                loss_value = self.__train_engine.value()
                print('Training State [epoch = {} loss = {:.8f}]'.format(epoch + 1, loss_value))
                if loss_value < loss_threshold:
                    break

    def predict(self, data):
        self.__predict_engine.symbol = self.__current_symbol
        self.__predict_engine.bind = {self.__input_symbol: data}
        predict_data = self.__predict_engine.value()
        return predict_data
