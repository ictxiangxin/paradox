import time
import collections
import numpy
from paradox.kernel.symbol import Variable
from paradox.kernel.engine import Engine
from paradox.kernel.optimizer import Optimizer, GradientDescentOptimizer
from paradox.neural_network.connection import ConnectionLayer, Connection
from paradox.neural_network.activation import ActivationLayer, Activation
from paradox.neural_network.loss import LossLayer, Loss


optimizer_map = {
    'gd': GradientDescentOptimizer,
}


def register_optimizer(name: str, optimizer: Optimizer):
    optimizer_map[name.lower()] = optimizer


class Network:
    def __init__(self):
        self.__layer = []
        self.__input_symbol = Variable(name='InputSymbol')
        self.__current_symbol = self.__input_symbol
        self.__current_output = None
        self.__current_weight = None
        self.__current_bias = None
        self.__variables = []
        self.__data = None
        self.__optimizer = None
        self.__loss = None
        self.__train_engine = Engine()
        self.__predict_engine = Engine()

    def add(self, layers):
        if isinstance(layers, collections.Iterable):
            for layer in layers:
                self.__add(layer)
        else:
            self.__add(layers)

    def __add(self, layer):
        if isinstance(layer, ConnectionLayer):
            self.__add_connection(layer)
        elif isinstance(layer, Connection):
            self.__add_connection(layer.connection_layer())
        elif isinstance(layer, ActivationLayer):
            self.__add_activation(layer)
        elif isinstance(layer, Activation):
            self.__add_activation(layer.activation_layer())
        else:
            raise ValueError('Invalid layer type: {}'. format(type(layer)))

    def __add_connection(self, layer: ConnectionLayer):
        weight, bias = layer.weight_bias(self.__current_output)
        self.__variables.append(weight)
        self.__variables.append(bias)
        self.__current_weight = weight
        self.__current_bias = bias
        self.__current_symbol = weight @ self.__current_symbol + bias
        self.__current_output = layer.output_dimension()

    def __add_activation(self, layer: ActivationLayer):
        self.__current_symbol = layer.activation_function(self.__current_symbol)
        self.__current_weight.value = layer.weight_initialization(self.__current_weight.value.shape)
        self.__current_bias.value = numpy.random.normal(0, 1, self.__current_bias.value.shape)

    def get_symbol(self):
        return self.__current_symbol

    def optimizer(self, optimizer_object, *args, **kwargs):
        if isinstance(optimizer_object, str):
            name = optimizer_object.lower()
            if name in optimizer_map:
                self.__optimizer = optimizer_map[name](*args, **kwargs)
            else:
                raise ValueError('No such optimizer: {}'.format(name))
        elif isinstance(optimizer_object, Optimizer):
            self.__optimizer = optimizer_object
        else:
            raise ValueError('Invalid optimizer type: {}'.format(type(optimizer_object)))

    def loss(self, loss_object):
        if isinstance(loss_object, str):
            self.__loss = Loss(loss_object).loss_layer()
        elif isinstance(loss_object, LossLayer):
            self.__loss = loss_object
        elif isinstance(loss_object, Loss):
            self.__loss = loss_object.loss_layer()
        else:
            raise ValueError('Invalid loss type: {}'.format(type(loss_object)))

    def train(self, data, target, epochs: int=10000, loss_threshold: float=0.001, state_cycle: int=100):
        loss = self.__loss.loss_function(self.__current_symbol, target)
        self.__train_engine.symbol = loss
        self.__train_engine.variables = self.__variables
        self.__train_engine.bind = {self.__input_symbol: data}
        start_time = time.time()
        cycle_start_time = time.time()
        try:
            for epoch in range(epochs):
                self.__optimizer.minimize(self.__train_engine)
                if (epoch + 1) % state_cycle == 0:
                    speed = state_cycle / (time.time() - cycle_start_time)
                    cycle_start_time = time.time()
                    loss_value = self.__train_engine.value()
                    print('Training State [epoch = {}/{}, loss = {:.8f}, speed = {:.2f}(epochs/s)'.format(
                        epoch + 1,
                        epochs,
                        loss_value,
                        speed))
                    if loss_value < loss_threshold:
                        print('Touch loss threshold: {} < {}'.format(loss_value, loss_threshold))
                        break
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
        print('Training Complete [{}]'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

    def predict(self, data):
        self.__predict_engine.symbol = self.__current_symbol
        self.__predict_engine.bind = {self.__input_symbol: data}
        predict_data = self.__predict_engine.value()
        return predict_data
