import torch
import numpy as np
import matplotlib.pyplot as plt

# 数据提取出来的x_data和y_data都得是矩阵
xy = np.loadtxt('E:\PyTorch深度学习实践\Code\diabetes.csv.gz',delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:,[-1]])
# torch.Size([759, 8]) torch.Size([759, 1]) 最后一列提出来的是矩阵形式

# y_data = torch.from_numpy(xy[:, -1])
# torch.Size([759, 8]) torch.Size([759]) 这种情况下拿出来的是向量

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activation_sigmoid = torch.nn.Sigmoid()
        self.activation_relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.activation_relu(self.linear1(x))
        x = self.activation_relu(self.linear2(x))
        x = self.activation_sigmoid(self.linear3(x))
        return x
    
model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
cost_list = []

for epoch in range(1000000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print('epoch=', epoch, loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 准确率测试都在整个样本集上完成的！
    # 在训练过程中定期地评估模型的性能，通常用于监控模型的训练过程
    if epoch%100000 == 99999:
        
        # 使用了阈值为0.5来将模型的输出（即预测的概率）转换为二元分类的标签。如果模型的输出概率大于等于0.5，则将其划分为标签1；如果小于0.5，则将其划分为标签0
        y_pred_label = torch.where(y_pred>=0.5,torch.tensor([1.0]),torch.tensor([0.0]))

        # 首先比较了模型预测的标签 y_pred_label 和真实的标签 y_data 是否相等，返回了一个布尔值的张量，其中 True 表示模型预测正确，False 表示模型预测错误
        # 计算布尔张量中 True 的数量，也就是模型正确预测的样本数量
        # 将计算得到的正确样本数量转换为一个单个的数字
        # 将正确样本数量除以总样本数量，得到准确率
        # acc 将包含模型在当前周期的训练集上的准确率
        acc = torch.eq(y_pred_label, y_data).sum().item()/y_data.size(0)
        print("loss = ",loss.item(), "acc = ",acc)
    epoch_list.append(epoch)
    cost_list.append(loss.item())

# 如果想查看某些层的参数，以神经网络的第一层参数为例，可按照以下方法进行。
'''
# 第一层的参数：
layer1_weight = model.linear1.weight.data
layer1_bias = model.linear1.bias.data
print("layer1_weight", layer1_weight)
print("layer1_weight.shape", layer1_weight.shape)
print("layer1_bias", layer1_bias)
print("layer1_bias.shape", layer1_bias.shape)
'''

    
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.grid()
plt.show()