import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

points_sum = 500

x_data = []
y_data = []

for i in range(points_sum):
    x = np.random.normal(0, 2)
    y = x * 2 + 1 + np.random.normal(0, 2)
    x_data.append(x)
    y_data.append(y)
x_np = np.array(x_data)
y_np = np.array(y_data)

x = pd.Symbol(x_np, name='x')
y = pd.Symbol(y_np, name='y')
w = pd.Symbol(0, name='w')
b = pd.Symbol(1, name='b')

loss = pd.reduce_sum((w * x + b - y) ** 2)
loss_engine = pd.Engine(loss, [w, b])
optimizer = pd.GradientDescentOptimizer(0.0001)

for epoch in range(100):
    optimizer.minimize(loss_engine)
    loss_value = loss_engine.value()
    print('loss = {:.8f}'.format(loss_value))

w_value = pd.Engine(w).value()
b_value = pd.Engine(b).value()

plt.plot(x_data, y_data, 'ro', label='Data')
plt.plot(x_data, w_value * x_data + b_value)
plt.legend()
plt.show()
