import time
import numpy
from paradox.kernel.engine import Engine
from paradox.neural_network.connection import ConnectionLayer, BatchNormalization, Connection
from paradox.neural_network.convolutional_neural_network.layer import ConvolutionLayer, Convolution


class Plugin:
    network = None
    enable = True

    def bind_network(self, network):
        self.network = network

    def begin_training(self):
        pass

    def end_training(self):
        pass

    def begin_epoch(self):
        pass

    def end_epoch(self):
        pass

    def begin_iteration(self):
        pass

    def end_iteration(self):
        pass

    def begin_predict(self):
        pass

    def end_predict(self):
        pass


class TrainingStatePlugin(Plugin):
    def __init__(self, state_cycle: int=100):
        self.state_cycle = state_cycle
        self.start_time = None
        self.cycle_start_time = None

    def begin_training(self):
        self.start_time = time.time()
        self.cycle_start_time = self.start_time

    def end_training(self):
        print('Training Complete [{}]'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))))

    def begin_iteration(self):
        if self.network.iteration % self.state_cycle == 0:
            self.cycle_start_time = time.time()

    def end_iteration(self):
        if self.network.iteration % self.state_cycle == 0:
            speed = self.state_cycle / (time.time() - self.cycle_start_time)
            loss_value = self.network.engine.value()
            print('Training State [epoch = {}/{}, loss = {:.8f}, speed = {:.2f}(iterations/s)]'.format(
                self.network.epoch,
                self.network.epochs,
                loss_value,
                speed))


class VariableMonitorPlugin(Plugin):
    def __init__(self, layer_name=None, for_iteration: bool=True):
        self.__layer_name = None
        self.__for_iteration = for_iteration
        self.set_layer_name(layer_name)

    def get_layer_name(self):
        return self.__layer_name

    def set_layer_name(self, layer_name):
        if isinstance(layer_name, str):
            self.__layer_name = [layer_name]
        else:
            self.__layer_name = layer_name

    layer_name = property(get_layer_name, set_layer_name)

    def output_variables(self):
        if self.__layer_name is None:
            self.__layer_name = list(self.network.layer_name_map())
        for layer_name in self.__layer_name:
            layer = self.network.get_layer(layer_name)
            if isinstance(layer, ConnectionLayer) or isinstance(layer, Connection):
                layer = layer.connection_layer()
                weight, bias = layer.weight_bias()
                print('[{}]: Weight = \n{}'.format(layer_name, weight.value))
                print('[{}]: Bias = \n{}'.format(layer_name, bias.value))
            elif isinstance(layer, ConvolutionLayer) or isinstance(layer, Convolution):
                layer = layer.convolution_layer()
                kernel = layer.kernel()
                print('[{}]: Kernel = \n{}'.format(layer_name, kernel.value))

    def end_iteration(self):
        if self.__for_iteration:
            self.output_variables()

    def end_epoch(self):
        if not self.__for_iteration:
            self.output_variables()


class BatchNormalizationPlugin(Plugin):
    def __init__(self):
        self.__batch_normalization_layer = []

    def begin_training(self):
        for layer in self.network.layer_stack():
            if isinstance(layer, BatchNormalization):
                self.__batch_normalization_layer.append([layer, [], []])

    def end_epoch(self):
        for layer_tuple in self.__batch_normalization_layer:
            layer_tuple[1] = [numpy.mean(numpy.array(layer_tuple[1]), axis=0)]
            layer_tuple[2] = [numpy.mean(numpy.array(layer_tuple[2]), axis=0)]

    def end_iteration(self):
        for layer_tuple in self.__batch_normalization_layer:
            layer_mean_symbol, layer_variance_symbol = layer_tuple[0].normalization_symbol()
            normalization_engine = Engine(layer_variance_symbol)
            normalization_engine.bind = self.network.engine.bind
            layer_tuple[2].append(normalization_engine.value())
            layer_tuple[1].append(normalization_engine.value_cache[layer_mean_symbol])

    def begin_predict(self):
        value_cache = {}
        for layer_tuple in self.__batch_normalization_layer:
            layer_mean_symbol, layer_variance_symbol = layer_tuple[0].normalization_symbol()
            value_cache[layer_mean_symbol] = numpy.mean(numpy.array(layer_tuple[1]), axis=0)
            value_cache[layer_variance_symbol] = numpy.mean(numpy.array(layer_tuple[2]), axis=0)
        self.network.predict_engine.value_cache = value_cache


default_plugin = [
    ('Training State', TrainingStatePlugin(), True),
    ('Variable Monitor', VariableMonitorPlugin(), False),
    ('Batch Normalization', BatchNormalizationPlugin(), True),
]
