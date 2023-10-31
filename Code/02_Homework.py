import numpy as np
# For plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return w * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.0, 0.1):
    for b in np.arange(-2.0, 2.0, 0.1):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
        mse_list.append(l_sum/3)
        b_list.append(b)
    w_list.append(w)

mse = np.array(mse_list).reshape(40, 40)
b = np.array(b_list).reshape(40, 40)
w = np.array(w_list)


fig = plt.figure()  # 创建一个新的Matplotlib图形队形
ax = fig.add_subplot(111, projection='3d')  # 创建了一个三维坐标系，并将它添加到了图形对象中。111 表示一个 1x1 的网格，第一个子图，projection='3d' 指定了这是一个三维坐标系。
[w, b] = np.meshgrid(w, b[1])  # 使用 np.meshgrid 函数创建了两个网格，分别对应 w 和 b。w 是 w 的网格，b[1] 是 b 的第二行，也就是一个数列。这两者将会作为 x 和 y 坐标传递给 ax.plot_surface。
ax.plot_surface(w, b, mse)  # ax.plot_surface(w, b, mse) 绘制了一个三维曲面图，以 w 作为 x 坐标，b 作为 y 坐标，mse 作为 z 坐标
plt.show()  # 显示图形