import time
import collections
import numpy
from paradox.kernel.operator import Operator
from paradox.kernel.symbol import Variable
from paradox.kernel.engine import Engine
from paradox.kernel.optimizer import Optimizer, GradientDescentOptimizer
from paradox.neural_network.loss import LossLayer, Loss
from paradox.neural_network.connection import ConnectionLayer, Connection
from paradox.neural_network.activation import ActivationLayer, Activation
from paradox.neural_network.convolutional_neural_network.layer import ConvolutionLayer, PoolingLayer, UnpoolingLayer
from paradox.neural_network.convolutional_neural_network.layer import Convolution, Pooling, Unpooling


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
        self.__plugin = []

    def add(self, layers):
        if isinstance(layers, collections.Iterable):
            for layer in layers:
                self.__add(layer)
        else:
            self.__add(layers)

    def __add(self, layer):
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
        self.__current_symbol = Variable(operator=layer, inputs=[self.__current_symbol])

    def __add_connection(self, layer: ConnectionLayer):
        weight, bias = layer.weight_bias(self.__current_output)
        self.__variables.append(weight)
        self.__variables.append(bias)
        self.__current_weight = weight
        self.__current_bias = bias
        self.__current_symbol = self.__current_symbol @ weight + bias
        self.__current_output = layer.output_dimension()

    def __add_activation(self, layer: ActivationLayer):
        self.__current_symbol = layer.activation_function(self.__current_symbol)
        self.__current_weight.value = layer.weight_initialization(self.__current_weight.value.shape)
        self.__current_bias.value = layer.bias_initialization(self.__current_bias.value.shape)

    def __add_convolution(self, layer: ConvolutionLayer):
        self.__variables.append(layer.kernel())
        self.__current_symbol = layer.convolution_function()(self.__current_symbol, layer.kernel(), layer.mode())

    def __add_pooling(self, layer: PoolingLayer):
        self.__current_symbol = layer.pooling_function()(self.__current_symbol, layer.size(), layer.step())

    def __add_unpooling(self, layer: UnpoolingLayer):
        self.__current_symbol = layer.unpooling_function()(self.__current_symbol, layer.size(), layer.step())

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

    def train(self,
              data,
              target,
              epochs: int=10000,
              batch_size: int=0,
              loss_threshold: float=0.001,
              state_cycle: int=100):
        if data.shape[0] != target.shape[0]:
            raise ValueError('Data dimension not match target dimension: {} {}'.format(data.shape[0], target.shape[0]))
        data_scale = data.shape[0]
        if batch_size == 0:
            batch_size = data_scale
        loss, target_symbol = self.__loss.loss_function(self.__current_symbol, target[:batch_size], True)
        self.__train_engine.symbol = loss
        self.__train_engine.variables = self.__variables
        start_time = time.time()
        cycle_start_time = time.time()
        try:
            iteration = 0
            for epoch in range(epochs):
                for i in range(0, data_scale, batch_size):
                    self.__train_engine.bind = {self.__input_symbol: data[i: min([i + batch_size, data_scale])],
                                                target_symbol: target[i: min([i + batch_size, data_scale])]}
                    self.__optimizer.minimize(self.__train_engine)
                    iteration += 1
                    if iteration % state_cycle == 0:
                        speed = state_cycle / (time.time() - cycle_start_time)
                        cycle_start_time = time.time()
                        loss_value = self.__train_engine.value()
                        print('Training State [epoch = {}/{}, loss = {:.8f}, speed = {:.2f}(iteration/s), {}]'.format(
                            epoch + 1,
                            epochs,
                            loss_value,
                            speed,
                            self.run_plugin()))
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

    def add_plugin(self, plugin_function, output_format):
        self.__plugin.append((output_format, plugin_function))

    def run_plugin(self):
        output_format_list = []
        result_list = []
        for output_format, plugin_function in self.__plugin:
            output_format_list.append(output_format)
            result_list.append(plugin_function())
        return ', '.join(output_format_list).format(*result_list)
