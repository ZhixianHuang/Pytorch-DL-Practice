import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081,))
])

train_dataset = datasets.MNIST(root= 'E:\PyTorch深度学习实践\dataset', train=True, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root= 'E:\PyTorch深度学习实践\dataset', train=False, download=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class InceptionA(torch.nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)    
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)
        
        self.branch_Cov1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        
        self.branch_Con5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_Con5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
        
        self.branch_Con3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_Con3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch_Con3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
        
    def forward(self, x):
        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        branch_1x1 = self.branch_Cov1x1(x)
        
        branch_5x5 = self.branch_Con5x5_1(x)
        branch_5x5 = self.branch_Con5x5_2(branch_5x5)
        
        branch_3x3 = self.branch_Con3x3_1(x)
        branch_3x3 = self.branch_Con3x3_2(branch_3x3)
        branch_3x3 = self.branch_Con3x3_3(branch_3x3)
        
        outputs = [branch_pool, branch_1x1, branch_5x5, branch_3x3]
        return torch.cat(outputs, dim=1)
    
class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)
        
        self.Incep1 = InceptionA(in_channels=10)
        self.Incep2 = InceptionA(in_channels=20)
        
        self.activation_ReLU = torch.nn.ReLU()
        self.MaxPooL = torch.nn.MaxPool2d(2)
        
        self.linear1 = torch.nn.Linear(1408,256)
        self.linear2 = torch.nn.Linear(256, 64)
        self.linear3 = torch.nn.Linear(64, 10) 
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.activation_ReLU(self.MaxPooL(self.conv1(x)))
        x = self.Incep1(x)
        x = self.activation_ReLU(self.MaxPooL(self.conv2(x)))
        x = self.Incep2(x)
        x = x.view(batch_size, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using Device: {device}')
model.to(device)
 
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
    for epoch in range(10):
        train(epoch)
        acc = test_acc()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()