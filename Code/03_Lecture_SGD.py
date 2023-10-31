import random
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return w * x

def loss(x, y):
    return (forward(x) - y) ** 2

def cost(xs, ys):
    cost = 0
    for x_val, y_val in zip(xs, ys):
        loss_val = (forward(x_val) - y_val) ** 2    # Loss Function
        cost += loss_val    # Cost Function 是对Loss Function对每个数据求和除以数据容量                                
    return cost / len(xs)

def gradient(x, y):
    return 2 * x * (w * x - y)

epoch_list = []
cost_list_single = []
cost_list_double = []
cost_list_cost = []


# 使用单个SGD更新
for epoch in range(50):
    i = random.randint(0,2)
    loss_val = loss(x_data[i], y_data[i])
    w -= gradient(x_data[i], y_data[i]) * 0.005
    epoch_list.append(epoch)
    cost_val = cost(x_data,y_data)
    cost_list_single.append(w)
    print('epoch=', epoch, 'w=', w, 'cost=',loss_val)
    
# 使用两个loss取平均
w = 1.0
for epoch in range(50):
    nums = [0, 1, 2]
    rand_nums = random.sample(nums, 2)  # 随机选择两个索引
    loss_val_double = (loss(x_data[rand_nums[0]], y_data[rand_nums[0]]) + loss(x_data[rand_nums[1]], y_data[rand_nums[1]]) )/2  # 计算两个样本的损失均值
    w -= (gradient(x_data[rand_nums[0]], y_data[rand_nums[0]]) + gradient(x_data[rand_nums[1]], y_data[rand_nums[1]])) /2 * 0.005  # 更新权重，除以2是为了取均值，继承自loss
    cost_list_double.append(w)

'''
for epoch in range(1000):
    i1, i2 = random.sample(range(len(x_data)), 2)  # 随机选择两个索引
    x1, y1 = x_data[i1], y_data[i1]
    x2, y2 = x_data[i2], y_data[i2]

    loss_val = (loss(x1, y1) + loss(x2, y2)) / 2  # 计算两个样本的损失均值
    w -= (gradient(x1, y1) + gradient(x2, y2)) * 0.01 / 2  # 更新权重，除以2是为了取均值
    epoch_list.append(epoch)
    cost_val = cost(x_data, y_data)
    cost_list.append(cost_val)
    print('epoch=', epoch, 'w=', w, 'cost=', loss_val)
'''

# 使用三个loss取平均(也就是loss)
w = 1.0
for epoch in range(50):
    nums = [0, 1, 2]
    rand_nums = random.sample(nums, 3)
    loss_val_double = (loss(x_data[rand_nums[0]], y_data[rand_nums[0]]) + loss(x_data[rand_nums[1]], y_data[rand_nums[1]]) + loss(x_data[rand_nums[2]], y_data[rand_nums[2]]) )/ 3
    w -= (gradient(x_data[rand_nums[0]], y_data[rand_nums[0]]) + gradient(x_data[rand_nums[1]], y_data[rand_nums[1]]) + gradient(x_data[rand_nums[2]], y_data[rand_nums[2]])) /3 * 0.005
    cost_list_cost.append(w)
    
plt.plot(epoch_list,cost_list_single, color='orange', label='single_SGD')
plt.plot(epoch_list,cost_list_double, color='blue', label='double_SGD')
plt.plot(epoch_list,cost_list_cost, color='green', label='cost')
plt.legend(loc='upper right')

# 单个SGD拥有最高随机率 也就是最高的波动性

plt.ylabel('cost')
plt.xlabel('epoch')
plt.grid()
plt.show()
