# Paradox 小型深度学习框架

![Version](https://img.shields.io/badge/Version-0.2-blue.svg) ![Version](https://img.shields.io/badge/Python-3.5.0-green.svg) ![Version](https://img.shields.io/badge/Numpy-1.13.0-green.svg)

> 用python3和numpy实现一个简单的深度学习框架。

![LinearRegression](https://raw.githubusercontent.com/ictxiangxin/paradox/master/documentations/images/graph_example.png)

* [代码示例](examples)
* [使用文档](documentations)

## 依赖

| Name   | Version |
|--------|---------|
| Python | 3.5.0+  |
| Numpy  | 1.13.0+ |

## 开始

使用Paradox对![](http://latex.codecogs.com/gif.latex?y=kx+b)进行梯度计算，并输出`x`的梯度。
整个过程由图计算和自动求导完成。

```python
import paradox as pd

k = pd.Constant([[2, 3], [1, 1]], name='k')
b = pd.Constant([[7], [3]], name='b')
x = pd.Variable([[0], [0]], name='x')

y = k @ x + b

print(pd.Engine(y).gradient(x).value)
```

运行结果
```
[[ 3.]
 [ 4.]]
```

## 功能实现

* 图计算（Graph Computing）。
* 自动求导（Auto Gradient）。
* 代数系统。
* 梯度下降。
* 神经网络API。
* 卷积神经网络（Convolutional Neural Network）。

## 联系

|        |                         |
|--------|-------------------------|
| Author | ict                     |
| QQ     | 405340537               |
| E-mail | ictxiangxin@hotmail.com |
