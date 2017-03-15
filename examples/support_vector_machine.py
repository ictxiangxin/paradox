import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

# 每类随机生成点的个数。
points_sum = 100

c1_x = []
c1_y = []
c2_x = []
c2_y = []

# 分别在(0, 0)点附近和(8, 8)点附近生成2类随机数据。
for _ in range(points_sum):
    c1_x.append(np.random.normal(0, 2))
    c1_y.append(np.random.normal(0, 2))
    c2_x.append(np.random.normal(8, 2))
    c2_y.append(np.random.normal(8, 2))

# 定义符号。
c1 = pd.Constant([c1_x, c1_y], name='c1')
c2 = pd.Constant([c2_x, c2_y], name='c2')
W = pd.Variable([[1, 1], [1, 1]], name='w')
B = pd.Variable([[1], [1]], name='b')

# 定义SVM loss函数。
loss = pd.reduce_mean(pd.maximum(0, [[1, -1]] @ (W @ c1 + B) + 1) + pd.maximum(0, [[-1, 1]] @ (W @ c2 + B) + 1))

# 创建loss计算引擎，申明变量为W和B。
loss_engine = pd.Engine(loss, [W, B])

# 创建梯度下降optimizer。
optimizer = pd.GradientDescentOptimizer(0.01)

# 迭代至多1000次最小化loss。
for epoch in range(1000):
    optimizer.minimize(loss_engine)
    loss_value = loss_engine.value()
    print('loss = {:.8f}'.format(loss_value))
    if loss_value < 0.001:  # loss阈值。
        break

# 获取W和B的训练结果。
w_data = pd.Engine(W).value()
b_data = pd.Engine(B).value()

# 计算分类直线的斜率和截距。
k = (w_data[1, 0] - w_data[0, 0]) / (w_data[0, 1] - w_data[1, 1])
b = (b_data[1, 0] - b_data[0, 0]) / (w_data[0, 1] - w_data[1, 1])

# 绘制图像。
plt.title('Paradox implement Linear SVM')
plt.plot(c1_x, c1_y, 'ro', label='Category 1')
plt.plot(c2_x, c2_y, 'bo', label='Category 2')
plt.plot([-5, 15], k * np.array([-5, 15]) + b, 'y', label='SVM')
plt.legend()
plt.show()
