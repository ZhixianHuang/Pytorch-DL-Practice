import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return w * x

def cost(xs, ys):
    cost = 0
    for x_val, y_val in zip(xs, ys):
        loss_val = (forward(x_val) - y_val) ** 2    # Loss Function
        cost += loss_val    # Cost Function 是对Loss Function对每个数据求和除以数据容量                                
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x_val, y_val in zip(xs, ys):
        grad += 2 * x_val *(w * x_val - y_val)  # 梯度公式 script page 3-13
                                                # 对Cost Function求梯度
    return grad / len(xs)                       # 求和与除以数据容量是从Cost Function处继承的

print('Predict (before training)', 4, forward(4))

epoch_list = []
cost_list = []

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val    # 梯度更新 learning rate alpha = 0.01 
                            # grad_val 是梯度上升放下 负方向是下降方向
    print('epoch=', epoch, 'w=', w, 'cost=', cost_val)
    # print('epoch=', epoch, 'w={:.2f},cost={:.2f}'.format(w, cost_val))
    epoch_list.append(epoch)
    cost_list.append(w)

print('Predict (after training)', 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.grid()
plt.show()