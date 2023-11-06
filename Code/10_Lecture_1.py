import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root= 'E:\PyTorch深度学习实践\dataset', train=True, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root= 'E:\PyTorch深度学习实践\dataset', train=False, download=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(320, 128)
        self.linear2 = torch.nn.Linear(128, 32)
        self.linear3 = torch.nn.Linear(32, 10)
        self.activation_ReLU = torch.nn.ReLU()
        
    def forward(self, x):
        # 为什么这里需要重新获取batch_size?
        # 从MINST获取的输入为(batch_size, 1, 28, 28)
        # 经过卷积网络后为(batch, 20, 4, 4)
        # 想要传入全连接网络的话需要传入的是一个矩阵(Lecture 09 传入为(N, 784))
        # 所以这里的传入是(batch_size, 320) 其中320可以通过自动计算得到，用-1占用
        # 计算原理 x = batch_size* 20* 4* 4 / batch_size (x = 20* 4* 4)
        # 用x.size(0)获取当前批次的样本数量 大多数的mini-batch的size都是预定义的64
        # 只有极少数是最后被抽取的batch会不等于64，需要用这种方式获取
        batch_size = x.size(0)
        x = self.pooling(self.activation_ReLU(self.conv1(x)))
        x = self.pooling(self.activation_ReLU(self.conv2(x)))
        
        # 为什么要使用动态的batch_size而不是固定的64
        # e.g. fullbatch = 1000, batch_size=64, 会剩下最后一个batch的batch_size=40 不满64
        # 最后一个batch只有40个样本 如果强制改成64，就不再可以传入全连接层(320个输出，128个输出)
        x = x.view(batch_size, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
        
'''
09_Lecture_GPU
class Net(torch.nn.Module):
    ...
    def forward(self, x):
        # 为什么这里不需要？
        # 输入是(batch_size, 1, 28, 28) 
        # 所以已知输出的矩阵特征为784 
        # batch_size是用自动计算的方式获得的
        # 所以和上述的刚好相反 
        # 上述是已知batch_size 自动计算输出维度
        # 这个是已知输出维度 自动计算batch_size
        
        x = x.view(-1, 784)  # input_dim = (N, 784)
        ...
        return self.linear15(x)
'''        

model = Net()

# GPU需要将模型，数据全都传入GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using Device: {device}')
# 将模型移动到GPU上
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_index_i, data in enumerate(train_loader, 0):
        input, label = data
        
        # 将数据移动到 GPU 上
        input, label = input.to(device), label.to(device)
        y_pred = model(input)
        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()

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
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            label_pred = model(images)
            _, predicted = torch.max(label_pred, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct/ total))
    return correct/total    

if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    for epoch in range(30):
        train(epoch)
        acc = test_acc()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()