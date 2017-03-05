# Paradox 了解深度学习框架的原理

> 用python3和numpy实现一个简单的深度学习框架，了解流行框架的原理。

* 写了一些例子放在了`examples`文件夹下。
* 需要完善的功能和fix的bug还很多。
* 基本上实现了图计算和梯度的符号计算。

> 下一步在解决大部分bug后尝试做符号计算化简的工作。

## 一些例子

### 递归下降解线性方程组

x_1 + 2 * x_2 = 3
x_1 + 3 * x_2 = 4

x_1, x_2 初始化为 0, 0

```python
import paradox as pd

# 定义符号，A为方程系数矩阵，x为自变量，b为常数项。
A = pd.Symbol([[1, 2], [1, 3]], name='A')
x = pd.Symbol([0, 0], name='x')
b = pd.Symbol([3, 4], name='b')

# 使用最小二乘误差定义loss。
loss = pd.reduce_sum((A @ x - b) ** 2)

# 创建梯度下降optimizer
optimizer = pd.GradientDescentOptimizer(0.01)

# 创建loss的计算引擎，申明变量为x。
loss_engine = pd.Engine(loss, x)

# 迭代至多10000次最小化loss。
for epoch in range(10000):
    optimizer.minimize(loss_engine)
    loss_value = loss_engine.value()
    print('loss = {:.8f}'.format(loss_value))
    if loss_value < 0.0000001: # loss阈值。
        break

# 输出最终结果。
print('\nx =\n{}'.format(x.value))
```

代码输出为：
```
...
loss = 0.00000010
loss = 0.00000010
loss = 0.00000010

x =
[ 0.99886023  1.00044064]
```

### 线性回归

```python
import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

points_sum = 200

x_data = []
y_data = []

# 生成y = 2 * x + 1直线附近的随机点。
for i in range(points_sum):
    x = np.random.normal(0, 2)
    y = x * 2 + 1 + np.random.normal(0, 2)
    x_data.append(x)
    y_data.append(y)
x_np = np.array(x_data)
y_np = np.array(y_data)

# 定义符号。
x = pd.Symbol(x_np, name='x')
y = pd.Symbol(y_np, name='y')
w = pd.Symbol(0, name='w')
b = pd.Symbol(1, name='b')

# 使用最小二乘误差。
loss = pd.reduce_sum((w * x + b - y) ** 2)

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
plt.plot(x_data, y_data, 'ro', label='Data')
plt.plot(x_data, w_value * x_data + b_value)
plt.legend()
plt.show()
```
