import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


# 线性回归 与普通回归区别在于
# 输出不再具有尺度上的对比意义 反而是概率对比
# 尺度上对比意思是，我的g.t. 是8 y_hat是 6, 两数值的差值可以直接反应与g.t.的差别(距离)
# 概率对比指的是 P(y='1') = 0.7, P(y='3') = 0.3 所以P='1'
# 损失函数需要修改 修改为交叉熵损失函数BCE 原理见视频31:15
# BCE(y_val(g.t.), y_pred) = - (y_val * log(y_pred) + (1 - y_val) * log(1 - y_pred))


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1, 1)  
        
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))  # 1. 为线性模块添加非线性激活函数Sigmoid 使得输出处于[0,1]满足概率分布
                                            # PyTorch 中的 torch.nn.functional 模块包含了许多神经网络的函数，包括各种激活函数、损失函数等。
        return y_pred
    

model = LogisticRegressionModel()

criterion = torch.nn.BCELoss()  # 损失函数改为二项交叉熵Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01)

epoch_list = []
loss_list = []

for epoch in range(10000):
    sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred = model(x_val)
        loss = criterion(y_pred, y_val)
        sum += loss.item()
        print('epoch=', epoch, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_list.append(epoch)
    loss_list.append(sum / 3)
    
    
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


plt.plot(epoch_list, loss_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.grid()
plt.show()

