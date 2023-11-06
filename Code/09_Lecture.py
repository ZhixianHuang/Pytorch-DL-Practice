import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(root= 'E:\PyTorch深度学习实践\dataset', train=True, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root= 'E:\PyTorch深度学习实践\dataset', train=False, download=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size= batch_size,  shuffle=False)

class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear11 = torch.nn.Linear(784, 512)
        self.linear12 = torch.nn.Linear(512, 256)
        self.linear13 = torch.nn.Linear(256, 128)
        self.linear14 = torch.nn.Linear(128, 64)
        self.linear15 = torch.nn.Linear(64, 10)
        self.activation_ReLU = torch.nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation_ReLU(self.linear11(x))
        x = self.activation_ReLU(self.linear12(x))
        x = self.activation_ReLU(self.linear13(x))
        x = self.activation_ReLU(self.linear14(x))
        return self.linear15(x)
    
model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01, momentum=True)

def train(epoch):
    running_loss = 0.0
    for batch_index_i, data in enumerate(train_loader, 0):
        input, label = data
        
        y_pred = model(input)
        loss = criterion(y_pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_index_i % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_index_i+1, running_loss/300))
            running_loss = 0.0
            

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            label_pred = model(images)
            _, predicted = torch.max(label_pred.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct/ total))
        
        
if __name__ == '__main__':
    for epoch in range(1000):
        
        
        train(epoch)
        test()