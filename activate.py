import numpy as np
import matplotlib.pyplot as plt

# 定义不同的激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# 生成输入数据
x = np.linspace(-10, 10, 100)

# 计算激活函数的输出
sigmoid_output = sigmoid(x)
tanh_output = tanh(x)
relu_output = relu(x)

# 绘图
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(3, 1, 1)
plt.title('Sigmoid Activation Function')
plt.plot(x, sigmoid_output, label='Sigmoid', color='b')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.ylim(-0.1, 1.1)
plt.grid()
plt.legend()

# Tanh
plt.subplot(3, 1, 2)
plt.title('Tanh Activation Function')
plt.plot(x, tanh_output, label='Tanh', color='g')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.ylim(-1.1, 1.1)
plt.grid()
plt.legend()

# ReLU
plt.subplot(3, 1, 3)
plt.title('ReLU Activation Function')
plt.plot(x, relu_output, label='ReLU', color='r')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.ylim(-1, 10)
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
