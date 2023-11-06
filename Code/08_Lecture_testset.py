from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


raw_data = np.loadtxt('Diabetes.csv.gz', delimiter=',', dtype=np.float32)
raw_data_x = raw_data[:, :-1]
raw_data_y = raw_data[:,[-1]]
X_train, X_test , Y_train, Y_test = train_test_split(raw_data_x, raw_data_y, test_size=0.2)

class DiabetesDataset(Dataset):
    def __init__(self, inputs, label) -> None:
        super().__init__()
        self.x_data = torch.from_numpy(inputs)
        self.y_data = torch.from_numpy(label)
        self.len = inputs.shape[0]
        
    def __getitem__(self, index) -> Any:
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
X_test, Y_test = torch.from_numpy(X_test), torch.from_numpy(Y_test)
train_dataset = DiabetesDataset(X_train, Y_train)
train_dataloader = DataLoader(dataset= train_dataset, batch_size=32, shuffle=True, num_workers=0)

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = torch.nn.Linear(8,16)
        self.linear2 = torch.nn.Linear(16,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.activation_Sigmoid = torch.nn.Sigmoid()
        self.activation_ReLu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.activation_ReLu(self.linear1(x))
        x = self.activation_ReLu(self.linear2(x))
        x = self.activation_Sigmoid(self.linear3(x))
        return x     
        
model = Model()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)

epoch_list = []
cost_list = []

if __name__ == '__main__':

    for epoch in range(1000):
        sum = 0
        for i, (inputs, label) in enumerate(train_dataloader, 0):
            y_pred = model(inputs)
            loss =criterion(y_pred, label)
            sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_list.append(epoch)
        cost_list.append((sum / (train_dataset.len //32)+1))
    
    y_pred_test = model(X_test)
    y_pred_test_label = torch.where(y_pred_test>=0.5, torch.tensor([1.0]), torch.tensor([0.0]))
    acc = torch.eq(y_pred_test_label, Y_test).sum().item() / Y_test.size(0)
    print('ACC=', acc)
    
    
    plt.plot(epoch_list, cost_list)
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.grid()
    plt.show()
            
    
