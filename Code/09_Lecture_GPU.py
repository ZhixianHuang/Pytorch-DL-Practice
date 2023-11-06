import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# batch_size 设置为超参
batch_size = 64

# 神经网络结构设置
# 分成了两部分 第一部分是将输入矩阵转换为向量
# 第二部分是神经网路的构造 底层输入784个元素 H(1)隐藏层输出为512个神经元，依次类推
# 最终输出层首先不需要非线性激活函数
# 其次最终输出为10个标签 0-10 所以最终输出层为10个元素 直接return

# 第一部分 类似一个PipeLine 对所有接收到的数值进行预设定(此处为转变为Pytorch向量+归一化)操作
# 输入是 28*28*1 (Height*Width*Channel) 而 pytorch接受的变量是(Channel*Height*Width)
# 所以原来的输入是(H*W*C) = (N, 28, 28, 1) 
# Pytorch的输入是(1, 28, 28)
# 神经网络期望输入比较小，最好处于[0, 1],最好满足正态分布，对训练有帮助
transform = transforms.Compose([
    transforms.ToTensor(),  # 转变为图像张量
    transforms.Normalize((0.1307,),(0.3081,))  # 进行归一化，0.1307是MNIST数据集的平均值，0.3801是数据集的标准差
    
])

# dataset preperation
train_dataset = datasets.MNIST(root= 'E:\PyTorch深度学习实践\dataset', train=True, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root= 'E:\PyTorch深度学习实践\dataset', train=False, download=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False)


# 第二部分
class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear11 = torch.nn.Linear(784, 512)
        self.linear12 = torch.nn.Linear(512, 256)
        self.linear13 = torch.nn.Linear(256, 128)
        self.linear14 = torch.nn.Linear(128, 64)
        self.linear15 = torch.nn.Linear(64, 10)
        self.activation_ReLU = torch.nn.ReLU()
        self.activation_Sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        # 接收到的输入是(N, 1, 28, 28)
        # 神经网络要求的是(N, xx)的二阶的N行矩阵
        # 所以改变张量的形状将28*28 拍扁成为一个向量
        # 第一个数-1代表自动计算N的值
        # 计算步骤 为 x = N * 28 * 28 / 784 自动计算N的值输入
        x = x.view(-1, 784)  # input_dim = (N, 784)
        x = self.activation_ReLU(self.linear11(x))
        x = self.activation_ReLU(self.linear12(x))
        x = self.activation_ReLU(self.linear13(x))
        x = self.activation_Sigmoid(self.linear14(x))
        return self.linear15(x)
    
model = Net()

# 损失函数使用多标签的 交叉熵损失函数+独热编码
# 所有的标签概率都处于(0,1)之间 并且总和为1 且相互之间能产生抗衡作用
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01, momentum=0.5)

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: {device}')

# 将模型移动到 GPU 上
model = model.to(device)

def train(epoch):
    running_loss = 0.0
    for batch_index_i, data in enumerate(train_loader, 0):
        input, label = data
        
        # 将数据移动到 GPU 上
        input, label = input.to(device), label.to(device)
        
        # 前向传播 计算损失
        y_pred = model(input)
        loss = criterion(y_pred, label)
        
        # 梯度清空在后向传播之前就可以 只要在梯度更新之前将梯度清零就可以了 否则上次的梯度就会被累加
        optimizer.zero_grad()
        # 后向传播
        loss.backward()
        # 梯度更新
        optimizer.step()
        
        running_loss += loss.item()
        if batch_index_i % 300 == 299:
            # epoch 和 batch_index_i 都是从0开始计算, 每300个minibatch打印当前的epoch和minibatch索引
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_index_i+1, running_loss/300))
            running_loss = 0.0

def test_acc():
    correct = 0
    total = 0
    # 以下的计算不用计算梯度
    with torch.no_grad():
        for data in test_loader:
            # images是输入
            # labels是真值
            images, labels = data
            
            # 将数据移动到 GPU 上
            images, labels = images.to(device), labels.to(device)
            
            # label_pred 是当前模型的输出
            # 模型的输出是(, 10)
            label_pred = model(images)
            
            # torch.max(label_pred.data,dim=1)
            # 输出是沿着axis=1的方向选出这十个值中的最大值,以及它的索引位置。也就是 return maximum, index
            # _, predicted = maximum, index 
            # _为占位符，由于我们会从这10个值中选出最大值所对应的标签，所以我们只需要索引，将索引定义为predicted           
            _, predicted = torch.max(label_pred, dim=1)
            
            # 由于每次dataloader会提取一个mini-batch
            # labels来自于一个mini-batch所以labels.size(0)就是当前mini-batch的size
            # total作为累加器，计算了一个full batch的size
            total += labels.size(0)
            
            # 计算其中相等的标签的个数，累加，提取为浮点数，对于所有的mini-batch累加得到full batch的标签
            # 除以full batch的size就得到了准确率
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct/ total))
        

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test_acc()
