import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

# 随机生成点的个数。
points_sum = 200

x_data = []
y_data = []

# 生成y = 2 * x + 1直线附近的随机点。
for _ in range(points_sum):
    x = np.random.normal(0, 2)
    y = x * 2 + 1 + np.random.normal(0, 2)
    x_data.append(x)
    y_data.append(y)
x_np = np.array(x_data)
y_np = np.array(y_data)

# 定义符号。
X = pd.Constant(x_np, name='x')
Y = pd.Constant(y_np, name='y')
w = pd.Variable(0, name='w')
b = pd.Variable(1, name='b')

# 使用最小二乘误差。
loss = pd.reduce_sum((w * X + b - Y) ** 2)

# 创建loss计算引擎，申明变量为w和b。
loss_engine = pd.Engine(loss, [w, b])

# 梯度下降optimizer。
optimizer = pd.GradientDescentOptimizer(0.0001)

# 迭代100次最小化loss。
for epoch in range(100):
    optimizer.minimize(loss_engine)
    loss_value = loss_engine.value()
    print('loss = {:.8f}'.format(loss_value))

# 获取w和b的训练值。
w_value = pd.Engine(w).value()
b_value = pd.Engine(b).value()

# 绘制图像。
plt.title('Paradox implement Linear Regression')
plt.plot(x_data, y_data, 'ro', label='Data')
plt.plot(x_data, w_value * x_data + b_value, label='Regression')
plt.legend()
plt.show()
