import numpy as np
import matplotlib.pyplot as plt
import paradox as pd

# sin函数采样点数。
points_number = 30

# 生成训练数据。
c_x = np.arange(points_number).reshape([points_number, 1])
c_y = np.sin(c_x / 5)

# 构造1x32x1回归神经网络（最后一层不激活）。
model = pd.nn.Network()
model.add(pd.nn.Dense(64, input_dimension=1))  # 1维输入32维输出的全连接层。
model.add(pd.nn.Activation('tanh'))  # 使用tanh激活函数。
model.add(pd.nn.Dense(1))
model.loss('mse')

# 使用Adam下降优化器。
model.optimizer('adaptive moment estimation', rate=0.001, decay=0.9, square_decay=0.999, consistent=True)

# 执行训练。
model.train(c_x, c_y, epochs=20000)

# 生成预测数据。
x = np.arange(-5, 35, 0.1)
y = model.predict(x.reshape([x.shape[0], 1]))

# 绘制图像。
plt.title('1x64x1 Regression Neural Network')
plt.plot(c_x, c_y, 'ro', label='Sin(x)')
plt.plot(x, y, 'b', label='Regression')
plt.legend()
plt.show()
