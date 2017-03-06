import paradox as pd

A = pd.Constant([[1, 2], [1, 3]], name='A')
x = pd.Variable([0, 0], name='x')
b = pd.Constant([3, 4], name='b')

loss = pd.reduce_sum((A @ x - b) ** 2)

optimizer = pd.GradientDescentOptimizer(0.01)
loss_engine = pd.Engine(loss, x)
for epoch in range(10000):
    optimizer.minimize(loss_engine)
    loss_value = loss_engine.value()
    print('loss = {:.8f}'.format(loss_value))
    if loss_value < 0.0000001:
        break
print('\nx =\n{}'.format(x.value))
