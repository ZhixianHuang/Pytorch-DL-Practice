import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3801, ))
])

train_dataset = datasets.MNIST(root='E:\PyTorch深度学习实践\dataset', train=True, download=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = datasets.MNIST(root='E:\PyTorch深度学习实践\dataset', train=False, download=False, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels = in_channels

        self.Conv_1 = torch.nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
        # self.Conv_2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.activation_ReLU = torch.nn.ReLU()
        
    def forward(self, x):
        y = self.activation_ReLU(self.Conv_1(x))
        y = self.Conv_1(y)
        return self.activation_ReLU(x + y)
    
class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 1*28*28 →
        self.Conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.Conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.ResBlock_1 = ResidualBlock(in_channels=16)
        self.ResBlock_2 = ResidualBlock(in_channels=32)
                
        self.activation_ReLU = torch.nn.ReLU()
        self.Linear1 = torch.nn.Linear(512, 128)
        self.Linear2 = torch.nn.Linear(128, 32)
        self.Linear3 = torch.nn.Linear(32, 10)
    
    def forward(self, x):
        in_size = x.size(0)  # 1*28*28
        #         16*12*12             ←          16*24*24
        x = self.MaxPool(self.activation_ReLU(self.Conv1(x)))
        #         16*12*12
        x = self.ResBlock_1(x)
        #         32*4*4               ←             32*8*8
        x = self.MaxPool(self.activation_ReLU(self.Conv2(x)))
        #         32*4*4
        x = self.ResBlock_2(x)
        #         in_size, 512
        x = x.view(in_size, -1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x
    
model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i%300 == 299:
            print("{:d}{:5d} loss: {:.3f}".format(epoch+1, i+1, running_loss/300))
            running_loss = 0.0
            
def test_acc():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            y_pred_labels = model(images)
            
            _,predicted_labels = torch.max(y_pred_labels, dim=1)
            total +=labels.size(0)
            correct += (predicted_labels == labels).sum().item()
    print('Accuarcy on test set :{:.2f} %'.format(100* correct/ total))
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
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.show()
            

            
        