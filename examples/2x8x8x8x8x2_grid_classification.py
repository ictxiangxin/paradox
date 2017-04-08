import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

# 每类随机生成点的个数。
points_sum = 50

# 调用paradox的数据生成器生成3x3网格状数据。
data = pd.data.grid_2d(points_sum, raw=3, column=3)

# 组合数据。
c_x = data[0][0] + data[1][0]
c_y = data[0][1] + data[1][1]

# 定义每个点的分类类别。
classification = [0] * len(data[0][0]) + [1] * len(data[1][0])

# 调用高层API生成2x8x8x8x8x2的网络，5层网络。
model = pd.nn.Network()
model.add(pd.nn.Dense(8, input_dimension=2))  # 2维输入8维输出的全连接层。
model.add(pd.nn.Activation('tanh'))  # 使用tanh激活函数。
model.add(pd.nn.Dense(8))
model.add(pd.nn.Activation('tanh'))
model.add(pd.nn.Dense(8))
model.add(pd.nn.Activation('tanh'))
model.add(pd.nn.Dense(8))
model.add(pd.nn.Activation('tanh'))
model.add(pd.nn.Dense(2))
model.add(pd.nn.Activation('tanh'))
model.loss('softmax')  # 使用softmax loss。

# 使用梯度下降优化器。
model.optimizer('gd', rate=0.0003)

# 执行训练。
model.train(np.array([c_x, c_y]).transpose(), classification, epochs=30000)

# 设置网格密度为0.1。
h = 0.1

# 生成预测采样点网格。
x, y = np.meshgrid(np.arange(np.min(c_x) - .1, np.max(c_x) + .1, h), np.arange(np.min(c_y) - .1, np.max(c_y) + .1, h))

# 生成采样点预测值。
z = model.predict(np.array([x.ravel(), y.ravel()]).transpose()).argmax(axis=1).reshape(x.shape)

# 绘制图像。
plt.title('2x8x8x8x8x2 Grid Classification')
plt.plot(data[0][0], data[0][1], 'ro', label='Category 1')
plt.plot(data[1][0], data[1][1], 'bo', label='Category 2')
plt.contourf(x, y, z, 2, cmap='RdBu', alpha=.6)
plt.legend()
plt.show()
