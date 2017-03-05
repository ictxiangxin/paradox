import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

points_sum = 200

x_data = []
y_data = []

for i in range(points_sum):
    x = np.random.normal(0, 2)
    y = x * 2 + 1 + np.random.normal(0, 2)
    x_data.append(x)
    y_data.append(y)
x_np = np.array(x_data).reshape([1, points_sum])
y_np = np.array(y_data).reshape([1, points_sum])

sym_x = pd.Symbol(x_np, name='x')
sym_y = pd.Symbol(y_np, name='y')
sym_w = pd.Symbol([[0]], name='w')
sym_b = pd.Symbol([[1]], name='b')
I = pd.Symbol(np.ones([points_sum, 1]), name='I')

loss = pd.reduce_sum((sym_w @ sym_x + sym_b @ pd.transpose(I) - sym_y) ** 2)
loss_engine = pd.Engine(loss, [sym_w, sym_b])
optimizer = pd.GradientDescentOptimizer(0.0001)

for epoch in range(100):
    optimizer.minimize(loss_engine)
    loss_value = loss_engine.value()
    print('loss = {:.8f}'.format(loss_value))

plt.plot(x_data, y_data, 'ro', label='Data')
plt.plot(x_data, (pd.Engine(sym_w).value() * x_data + pd.Engine(sym_b).value())[0])
plt.legend()
plt.show()
