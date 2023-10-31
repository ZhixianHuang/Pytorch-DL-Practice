import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]  # 输入
y_data = [2.0, 4.0, 6.0]  # 输出
                          # 同索引对应一组输入输出

def forward(x):
    return x * w  # 给定模型计算预测值 输入*权重

def loss(x, y):  # Loss Function
    y_pred = forward(x)  
    return (y_pred - y) * (y_pred - y)  # 计算Loss （y_hat预测值 - y 真值）^2 

w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):  # 预测权重出现在此范围内 于是在该范围内采样 对于每一次采样进行如下操作
                                    # 内建range()函数只接受整型参数，需求是一个浮点数序列
    print('w=', w)                  # 打印当前权重
    l_sum = 0                       # 清空loss
    for x_val, y_val in zip(x_data, y_data):  # 对于当前权重下采样的数据  x_val:输入  y_val:真值
                                              # zip() 函数用于将多个可迭代对象的元素一一对应地打包成一个元组的序列。
                                              # 在这段代码中，zip(x_data, y_data) 将 x_data 和 y_data 中对应位置的元素一一配对，形成一个元组的序列
                                              # 增加代码可读性，不使用zip可以使用 for i in range(len(x_data)): x(y)_val = x(y)_data[i]
        y_pred_val = forward(x_val)           # 计算预测值 y_pred_val:根据输入*权重得到预测值
        loss_val = loss(x_val, y_val)         # 与真值比较计算损失
        l_sum += loss_val                     # 对每个数据的Loss求和（尚未得到Cost function，除以数据容量）
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)            # 打印当前COS Mean Square Error是sigma(Loss)/3
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()