import torch 
import numpy as np
# 需要使用mini-batch分批训练需要使用到Dataset和DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activation_Sigmoid = torch.nn.Sigmoid()
        self.activation_ReLu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.activation_ReLu(self.linear1(x))
        x = self.activation_ReLu(self.linear2(x))
        x = self.activation_Sigmoid(self.linear3(x))
        return x        
        

model = Model()

# 全新的设置数据集的方式
# 以前是将数据提取出来分别放进两个torch中，现在依旧采用这种方法 但是依赖于继承Dataset

# Dataset是一个抽象类 意味着Dataset不可以被直接实例化，只能被其他类继承
# 但是需要重写init，geiitem，len的魔法函数
class DiabetesDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        # 与之前准备数据的方式相同 将数据准备为numpy数组 并从numpy数组中的对应位置抽取数据
        xy = np.loadtxt('Diabetes.csv.gz', delimiter=',', dtype=np.float32)
        # 最后一行为标签，之前均为数据
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        
        # 准备魔法函数__len__的返回值
        # 第0维是数据容量(样本个数)
        # 第1维是特征个数
        self.len = xy.shape[0]

    # 将对应索引的特征从x_data中取出，将对应索引的标签从y_data中取出
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    # 
    def __len__(self):
        return self.len
    

dataset = DiabetesDataset()
# batch_size 为mini-batch的size
# shuffle 打乱fullbatch重新抽取mini-batch 使得每一次构成不相同
# num_workers
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


epoch_list = []
cost_list = []


if __name__ == '__main__':
    for epoch in range(10000):
        sum = 0
        # enumerate 是一个内置函数，它可以同时获取迭代器的索引和值，以此来遍历DataLoader中的所有mini-batch
        # 返回值中的索引给到i 数据给到data
        for i , data in enumerate(train_loader, 0):
            # 再从data中抽取 inputs和label
            inputs, label = data
            # 前向传播 更新损失
            y_pred = model(inputs)
            loss = criterion(y_pred, label)
            sum += loss.item()
            # 清空梯度，反向传播，更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_list.append(epoch)
        cost_list.append(sum / ((dataset.y_data.size(0) // 32)+1))
        if epoch%1000 == 999:
            y_pred = model(dataset.x_data)
            y_pred_label = torch.where(y_pred>=0.5, torch.tensor([1.0]), torch.tensor([0.0]))
                
            acc = torch.eq(y_pred_label, dataset.y_data).sum().item() / dataset.y_data.size(0)
            print("loss = ",loss.item(), "acc = ",acc)
            
            
        
      
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.grid()
plt.show()      