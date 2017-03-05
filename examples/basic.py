import paradox as pd

A = pd.Symbol([[1, 2], [1, 3]], name='A')
x = pd.Symbol([[0], [0]], name='x')
b = pd.Symbol([[3], [4]], name='b')

loss = pd.reduce_sum((A @ x - b) ** 2) / 2

e = pd.Engine(loss, x)
print('loss formula =\n{}\n'.format(loss))
print('loss =\n{}\n'.format(e.value()))

x_gradient = e.gradient(x)
print('x gradient formula =\n{}\n'.format(x_gradient))
print('x gradient =\n{}\n'.format(pd.Engine(x_gradient).value()))
