import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

# 每类随机生成点的个数。
points_sum = 100

# 在(0, 0)点附近生成一堆点然后以4为半径在周围生成一堆点构成2类随机数据。
c1_x, c1_y, c2_x, c2_y = [], [], [], []
for _ in range(points_sum):
    c1_x.append(np.random.normal(0, 1))
    c1_y.append(np.random.normal(0, 1))
    r = np.random.normal(4, 1)
    theta = np.random.normal(0, 2 * np.pi)
    c2_x.append(r * np.cos(theta))
    c2_y.append(r * np.sin(theta))
c_x = c1_x + c2_x
c_y = c1_y + c2_y

# 定义符号。
A = pd.Variable([c_x, c_y], name='A')
W1 = pd.Variable(np.random.random((4, 2)), name='W1')  # 输入层到隐含层的权重矩阵。
W2 = pd.Variable(np.random.random((2, 4)), name='W2')  # 隐含层到输出层的权重矩阵。
B1 = pd.Variable(np.random.random((4, 1)), name='B1')  # 隐含层的偏置。
B2 = pd.Variable(np.random.random((2, 1)), name='B2')  # 输出层的偏置。
K = pd.Constant([[-1] * points_sum + [1] * points_sum, [1] * points_sum + [-1] * points_sum])

# 构建2x4x2网络，使用ReLu激活函数。
model = pd.maximum(W2 @ pd.maximum(W1 @ A + B1, 0) + B2, 0)

# 使用SVM loss。
loss = pd.reduce_mean(pd.maximum(pd.reduce_sum(K * model, axis=0) + 1, 0))


# 创建loss计算引擎，申明变量为W1，W2，B1和B2。
loss_engine = pd.Engine(loss, [W1, W2, B1, B2])

# 创建梯度下降optimizer。
optimizer = pd.GradientDescentOptimizer(0.03)

# 迭代至多10000次最小化loss。
for epoch in range(10000):
    optimizer.minimize(loss_engine)
    if epoch % 100 == 0:  # 每100次epoch检查一次loss。
        loss_value = loss_engine.value()
        print('loss = {:.8f}'.format(loss_value))
        if loss_value < 0.001:  # loss阈值。
            break

# 创建预测函数。
predict = pd.where(pd.reduce_sum([[-1], [1]] * model, axis=0) < 0, -1, 1)

# 创建预测函数计算引擎。
predict_engine = pd.Engine(predict)

# 设置网格密度为0.1。
h = 0.1

# 生成预测采样点网格。
x, y = np.meshgrid(np.arange(np.min(c_x) - 1, np.max(c_x) + 1, h), np.arange(np.min(c_y) - 1, np.max(c_y) + 1, h))

# 绑定变量值。
predict_engine.bind = {A: [x.ravel(), y.ravel()]}

# 生成采样点预测值。
z = predict_engine.value().reshape(x.shape)

# 绘制图像。
plt.title('Paradox implement 2x4x2 Neural Network')
plt.plot(c1_x, c1_y, 'ro', label='Category 1')
plt.plot(c2_x, c2_y, 'bo', label='Category 2')
plt.contourf(x, y, z, 2, cmap='RdBu', alpha=.6)
plt.legend()
plt.show()
