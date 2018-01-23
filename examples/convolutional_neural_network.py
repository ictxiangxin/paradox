import os
import numpy as np
import paradox as pd

# 调用paradox的MNIST接口读取数据。
mnist_data = pd.data.MNIST('E:/mnist').read()

model_save_path = 'cnn.bin'
batch_size = 100


class AccuracyPlugin(pd.nn.Plugin):  # 创建一个输出精确度和预测值的插件。
    def __init__(self, data, label, index_reverse_map):
        self.data = data
        self.label = label
        self.index_reverse_map = index_reverse_map

    def end_iteration(self):  # 该函数会在每次迭代完成后被调用。
        predict = np.array([self.index_reverse_map[i] for i in np.array(np.argmax(self.network.predict(self.data), axis=1))])
        accuracy = len(predict[predict == self.label]) / len(self.label) * 100
        print('Test State [accuracy = {:.2f}%]\n{}'.format(accuracy, predict.reshape((10, 10))))


class AutoSavePlugin(pd.nn.Plugin):  # 创建每个epoch结束后保存模型的插件。
    def __init__(self, save_path):
        self.__save_path = save_path

    def end_epoch(self):  # 该函数会在每个epoch结束后被调用。
        pd.utils.save(model_save_path, self.network)


# 构建卷积神经网络。
if os.path.exists(model_save_path):  # 如果存在保存的模型则载入，否则重新创建。
    model = pd.utils.load(model_save_path)
else:
    model = pd.nn.Network()
    model.add(pd.cnn.Convolution2DLayer((4, 5, 5), 'valid', input_shape=(28, 28)))  # 使用4个5x5的卷积核。
    model.add(pd.cnn.AveragePooling2DLayer((2, 2), (2, 2)))  # 2x2的均值池化。
    model.add(pd.cnn.Convolution2DLayer((2, 3, 3), 'valid'))  # 使用2个3x3的卷积核。
    model.add(pd.cnn.MaxPooling2DLayer((2, 2), (2, 2)))  # 2x2的max池化。
    model.add(pd.nn.Dense(100))  # 接入100个神经元的全连接层。
    model.add(pd.nn.Activation('tanh'))  # tanh激活。
    model.add(pd.nn.Dense(50))
    model.add(pd.nn.Activation('tanh'))
    model.add(pd.nn.Dense(10))  # 接入到输出0~9数字的输出层。
    model.loss('softmax')

# 使用梯度下降
model.optimizer('adaptive moment estimation', rate=0.00001, decay=0.9, square_decay=0.999, consistent=True)

# 创建测试集分类数据。
lm, _, irm = pd.utils.generate_label_matrix(mnist_data['train_label'])

# 使用100张图片进行测试。
test_image = mnist_data['test_image'][:100]
test_label = mnist_data['test_label'][:100]

# 装入精度、预测值输出插件和模型保存插件。
model.add_plugin('Accuracy', AccuracyPlugin(test_image, test_label, irm))
model.add_plugin('Auto Save', AutoSavePlugin(model_save_path))
model.plugin('Training State').state_cycle = 1  # 修改Training State插件的参数为一次迭代输出一次结果。
model.regularization('l2', 0.01)  # 使用L2正则化。
# 执行训练。
model.train(mnist_data['train_image'], lm, epochs=10, batch_size=batch_size)
