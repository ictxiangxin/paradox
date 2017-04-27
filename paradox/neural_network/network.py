import collections
from functools import reduce
import numpy
from paradox.kernel.operator import Operator
from paradox.kernel.symbol import SymbolCategory, Symbol, Placeholder, spread
from paradox.kernel.engine import Engine
from paradox.kernel.optimizer import Optimizer, GradientDescentOptimizer
from paradox.neural_network.loss import LossLayer, Loss
from paradox.neural_network.connection import ConnectionLayer, Connection
from paradox.neural_network.activation import ActivationLayer, Activation
from paradox.neural_network.convolutional_neural_network.layer import ConvolutionLayer, PoolingLayer, UnpoolingLayer
from paradox.neural_network.convolutional_neural_network.layer import Convolution, Pooling, Unpooling
from paradox.neural_network.plugin import Plugin, TrainingStatePlugin


optimizer_map = {
    'gd': GradientDescentOptimizer,
}


def register_optimizer(name: str, optimizer: Optimizer):
    optimizer_map[name.lower()] = optimizer


class Network:
    def __init__(self):
        self.epoch = None
        self.iteration = None
        self.epochs = None
        self.batch_size = None
        self.engine = Engine()
        self.__layer = []
        self.__layer_name_map = collections.OrderedDict()
        self.__layer_number = 0
        self.__input_symbol = Placeholder(name='InputSymbol')
        self.__current_symbol = self.__input_symbol
        self.__current_output = None
        self.__current_weight = None
        self.__current_bias = None
        self.__variables = []
        self.__data = None
        self.__optimizer = None
        self.__loss = None
        self.__predict_engine = Engine()
        self.__plugin = collections.OrderedDict()
        self.load_default_plugin()

    def __valid_current_output(self):
        if self.__current_output is None:
            raise ValueError('Current output is None.')
        else:
            return self.__current_output

    def layer(self, name: str):
        if name in self.__layer_name_map:
            return self.__layer_name_map[name]
        else:
            raise ValueError('No such layer in Network: {}'.format(name))

    def add(self, layer, name=None):
        if isinstance(layer, collections.Iterable):
            for i, l in enumerate(layer):
                if name is not None and i < len(name):
                    self.__add(l, name[i])
                else:
                    self.__add(l)
        else:
            self.__add(layer, name)

    def __add(self, layer, name: str=None):
        self.__layer.append(layer)
        if name is None:
            name = 'layer_{}'.format(self.__layer_number)
        if name in self.__layer_name_map:
            raise ValueError('Layer name has contained in Network: {}'.format(name))
        else:
            self.__layer_name_map[name] = layer
        self.__layer_number += 1
        if isinstance(layer, Operator):
            self.__add_operator(layer)
        elif isinstance(layer, ConnectionLayer):
            self.__add_connection(layer)
        elif isinstance(layer, Connection):
            self.__add_connection(layer.connection_layer())
        elif isinstance(layer, ActivationLayer):
            self.__add_activation(layer)
        elif isinstance(layer, Activation):
            self.__add_activation(layer.activation_layer())
        elif isinstance(layer, ConvolutionLayer):
            self.__add_convolution(layer)
        elif isinstance(layer, Convolution):
            self.__add_convolution(layer.convolution_layer())
        elif isinstance(layer, PoolingLayer):
            self.__add_pooling(layer)
        elif isinstance(layer, Pooling):
            self.__add_pooling(layer.pooling_layer())
        elif isinstance(layer, UnpoolingLayer):
            self.__add_unpooling(layer)
        elif isinstance(layer, Unpooling):
            self.__add_unpooling(layer.unpooling_layer())
        else:
            raise ValueError('Invalid layer type: {}'. format(type(layer)))

    def __add_operator(self, layer: Operator):
        self.__current_symbol = Symbol(operator=layer, inputs=[self.__current_symbol], category=SymbolCategory.operator)

    def __add_connection(self, layer: ConnectionLayer):
        if layer.input_dimension is None:
            current_output = self.__valid_current_output()
            if not isinstance(current_output, int):
                self.__current_symbol = spread(self.__current_symbol, 1 - len(current_output))
                current_output = reduce(lambda a, b: a * b, current_output[1:])
            layer.input_dimension = current_output
        weight, bias = layer.weight_bias()
        self.__variables.append(weight)
        self.__variables.append(bias)
        self.__current_weight = weight
        self.__current_bias = bias
        self.__current_symbol = self.__current_symbol @ weight + bias
        self.__current_output = layer.output_dimension

    def __add_activation(self, layer: ActivationLayer):
        self.__current_symbol = layer.activation_function(self.__current_symbol)
        self.__current_weight.value = layer.weight_initialization(self.__current_weight.value.shape)
        self.__current_bias.value = layer.bias_initialization(self.__current_bias.value.shape)

    def __add_convolution(self, layer: ConvolutionLayer):
        kernel = layer.kernel()
        self.__variables.append(kernel)
        self.__current_symbol = layer.convolution_function()(self.__current_symbol, kernel, layer.mode)
        if layer.input_shape is None:
            layer.input_shape = self.__valid_current_output()
        self.__current_output = layer.get_output_shape()

    def __add_pooling(self, layer: PoolingLayer):
        self.__current_symbol = layer.pooling_function()(self.__current_symbol, layer.size, layer.step)
        if layer.input_shape is None:
            layer.input_shape = self.__valid_current_output()
        self.__current_output = layer.get_output_shape()

    def __add_unpooling(self, layer: UnpoolingLayer):
        self.__current_symbol = layer.unpooling_function()(self.__current_symbol, layer.size, layer.step)
        if layer.input_shape is None:
            layer.input_shape = self.__valid_current_output()
        self.__current_output = layer.get_output_shape()

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

    def train(self, data, target, epochs: int=10000, batch_size: int=0):
        data = numpy.array(data)
        target = numpy.array(target)
        self.epochs = epochs
        if data.shape[0] != target.shape[0]:
            raise ValueError('Data dimension not match target dimension: {} {}'.format(data.shape[0], target.shape[0]))
        data_scale = data.shape[0]
        target_symbol = None
        if batch_size != 0:
            loss, target_symbol = self.__loss.loss_function(self.__current_symbol, target[:batch_size], True)
        else:
            loss = self.__loss.loss_function(self.__current_symbol, target)
            self.engine.bind = {self.__input_symbol: data}
        self.engine.symbol = loss
        self.engine.variables = self.__variables
        try:
            self.iteration = 0
            self.run_plugin('begin_training')
            for epoch in range(self.epochs):
                self.epoch = epoch + 1
                self.run_plugin('begin_epoch')
                for i in ([0] if batch_size == 0 else range(0, data_scale, batch_size)):
                    if batch_size != 0:
                        self.engine.bind = {self.__input_symbol: data[i: min([i + batch_size, data_scale])],
                                            target_symbol: target[i: min([i + batch_size, data_scale])]}
                    self.iteration += 1
                    self.run_plugin('begin_iteration')
                    self.__optimizer.minimize(self.engine)
                    self.run_plugin('end_iteration')
                self.run_plugin('end_epoch')
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
        self.run_plugin('end_training')

    def predict(self, data):
        self.__predict_engine.symbol = self.__current_symbol
        self.__predict_engine.bind = {self.__input_symbol: data}
        predict_data = self.__predict_engine.value()
        return predict_data

    def load_default_plugin(self):
        default_plugin = [
            ('Training State', TrainingStatePlugin()),
        ]
        for name, plugin in default_plugin:
            self.add_plugin(name, plugin)

    def add_plugin(self, name: str, plugin: Plugin):
        self.__plugin[name] = plugin
        plugin.bind_network(self)

    def run_plugin(self, stage: str):
        for _, plugin in self.__plugin.items():
            if plugin.enable:
                getattr(plugin, stage)()

    def plugin(self, name: str):
        if name in self.__plugin:
            return self.__plugin[name]
        else:
            raise ValueError('No such plugin: {}'.format(name))
