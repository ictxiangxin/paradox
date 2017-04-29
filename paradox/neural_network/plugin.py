import time
from paradox.neural_network.connection import ConnectionLayer, Connection
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

    def end_iteration(self):
        if self.network.iteration % self.state_cycle == 0:
            speed = self.state_cycle / (time.time() - self.cycle_start_time)
            self.cycle_start_time = time.time()
            loss_value = self.network.engine.value()
            print('Training State [epoch = {}/{}, loss = {:.8f}, speed = {:.2f}(iterations/s)]'.format(
                self.network.epoch,
                self.network.epochs,
                loss_value,
                speed))


class VariableMonitorPlugin(Plugin):
    def __init__(self, layer_name: str, for_iteration: bool=True):
        self.__layer_name = layer_name
        self.__for_iteration = for_iteration

    def output_variables(self):
        layer = self.network.layer(self.__layer_name)
        layer = layer.connection_layer()
        if isinstance(layer, ConnectionLayer):
            weight, bias = layer.weight_bias()
            print('[{}]: Weight = \n{}'.format(self.__layer_name, weight.value))
            print('[{}]: Bias = \n{}'.format(self.__layer_name, bias.value))
        elif isinstance(layer, ConvolutionLayer):
            kernel = layer.kernel()
            print('[{}]: Kernel = \n{}'.format(self.__layer_name, kernel.value))

    def end_iteration(self):
        if self.__for_iteration:
            self.output_variables()

    def end_epoch(self):
        if not self.__for_iteration:
            self.output_variables()
