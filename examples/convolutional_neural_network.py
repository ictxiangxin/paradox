import numpy as np
import paradox as pd

mnist_data = pd.data.MNIST('E:/mnist').read()

batch_size = 100


class AccuracyPlugin(pd.nn.Plugin):
    def __init__(self, data, label, index_reverse_map):
        self.data = data
        self.label = label
        self.index_reverse_map = index_reverse_map

    def end_iteration(self):
        predict = np.array([self.index_reverse_map[i] for i in np.array(np.argmax(self.network.predict(self.data), axis=1))])
        accuracy = len(predict[predict == self.label]) / len(self.label) * 100
        print('Test State [accuracy = {:.2f}%]\n{}'.format(accuracy, predict.reshape((10, 10))))


model = pd.nn.Network()
model.add(pd.cnn.Convolution2DLayer((4, 5, 5), 'valid', input_shape=(None, 28, 28)))
model.add(pd.cnn.AveragePooling2DLayer((2, 2), (2, 2)))
model.add(pd.cnn.Convolution2DLayer((2, 3, 3), 'valid'))
model.add(pd.cnn.MaxPooling2DLayer((2, 2), (2, 2)))
model.add(pd.nn.Dense(100))
model.add(pd.nn.Activation('tanh'))
model.add(pd.nn.Dense(50))
model.add(pd.nn.Activation('tanh'))
model.add(pd.nn.Dense(10))
model.add(pd.nn.Activation('tanh'))
model.loss('softmax')

model.optimizer('gd', rate=0.0002, consistent=True)

lm, _, irm = pd.utils.generate_label_matrix(mnist_data['train_label'])
test_image = mnist_data['test_image'][:100]
test_label = mnist_data['test_label'][:100]


model.add_plugin('accuracy', AccuracyPlugin(test_image, test_label, irm))
model.plugin('Training State').state_cycle = 1

model.train(mnist_data['train_image'], lm, epochs=10, batch_size=batch_size)
