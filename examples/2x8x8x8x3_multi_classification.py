import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

# 每类随机生成点的个数。
points_sum = 100

# 调用paradox的数据生成器生成三螺旋的3类数据。
data = pd.data.helical_2d(100, 3, max_radius=2*np.pi)

# 组合数据。
c_x = data[0][0] + data[1][0] + data[2][0]
c_y = data[0][1] + data[1][1] + data[2][1]

# 定义每个点的分类类别。
classification = [0] * points_sum + [1] * points_sum + [2] * points_sum

# 调用高层API生成2x8x8x8x3的网络
model = pd.nn.Network()
model.add(pd.nn.Dense(8, input_dimension=2))  # 2维输入8维输出的全连接层。
model.add(pd.nn.Activation('tanh'))  # 使用tanh激活函数。
model.add(pd.nn.Dense(8))
model.add(pd.nn.Activation('tanh'))
model.add(pd.nn.Dense(8))
model.add(pd.nn.Activation('tanh'))
model.add(pd.nn.Dense(3))
model.add(pd.nn.Activation('tanh'))
model.loss('softmax')  # 使用softmax loss。

# 使用梯度下降优化器。
model.optimizer('gd', rate=0.0002)

# 执行训练。
model.train([c_x, c_y], classification, epochs=20000)

# 设置网格密度为0.1。
h = 0.1

# 生成预测采样点网格。
x, y = np.meshgrid(np.arange(np.min(c_x) - 1, np.max(c_x) + 1, h), np.arange(np.min(c_y) - 1, np.max(c_y) + 1, h))

# 生成采样点预测值。
z = model.predict([x.ravel(), y.ravel()]).argmax(axis=0).reshape(x.shape)

# 绘制图像。
plt.title('2x8x8x8x3 Multi-Classification')
plt.plot(data[0][0], data[0][1], 'bo', label='Category 1')
plt.plot(data[1][0], data[1][1], 'ro', label='Category 2')
plt.plot(data[2][0], data[2][1], 'go', label='Category 3')
plt.contourf(x, y, z, 3, cmap='brg', alpha=.6)
plt.legend()
plt.show()
