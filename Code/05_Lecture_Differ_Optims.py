import torch
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
criterion = torch.nn.MSELoss(reduction='sum')

def train_with_optimizer(optimizer, model):
    epoch_list = []
    loss_list = []
    for epoch in range(100):
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
    
    return epoch_list, loss_list
        
# Optimizer Adagrad
model = LinearModel()
optimzer_Adagrad = torch.optim.Adagrad(model.parameters(), lr=0.01)
epoch_list , loss_list_Adagrad = train_with_optimizer(optimzer_Adagrad, model)

# Optimzer Adam
model = LinearModel()
optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.01)
epoch_list, loss_list_Adam = train_with_optimizer(optimizer_Adam, model)

# Optimizer Adamax
model = LinearModel()
optimizer_Adamax = torch.optim.Adamax(model.parameters(), lr=0.01)
epoch_list, loss_list_Adamax = train_with_optimizer(optimizer_Adamax, model)

# Optimzer SGD
model = LinearModel()
optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.01)
epoch_list, loss_list_SGD = train_with_optimizer(optimizer_SGD, model)

# Optimzer ASGD
model = LinearModel()
optimizer_ASGD = torch.optim.ASGD(model.parameters(), lr=0.01)
epoch_list, loss_list_ASGD = train_with_optimizer(optimizer_ASGD, model)

# Optimzer Rmsprop
model = LinearModel()
optimizer_RMSprop = torch.optim.RMSprop(model.parameters(), lr=0.01)
epoch_list, loss_list_RMSprop = train_with_optimizer(optimizer_RMSprop, model)

# Optimzer Rprop
model = LinearModel()
optimizer_Rprop = torch.optim.Rprop(model.parameters(), lr=0.01)
epoch_list, loss_list_Rprop = train_with_optimizer(optimizer_Rprop, model)

# Optimzer Adadelta
model = LinearModel()
optimizer_Adadelta = torch.optim.Adadelta(model.parameters(), lr=0.01)
epoch_list, loss_list_Adadelta = train_with_optimizer(optimizer_Adadelta, model)


plt.plot(epoch_list, loss_list_Adagrad,color='red', label='Adagrad' )
plt.plot(epoch_list, loss_list_Adam, color='orange', label='Adam')
plt.plot(epoch_list, loss_list_Adamax, color='yellow', label='Adamax')
plt.plot(epoch_list, loss_list_SGD, color='green', label='SGD')
plt.plot(epoch_list, loss_list_ASGD, color='blue', label='ASGD')
plt.plot(epoch_list, loss_list_RMSprop, color='purple', label='RMSPROP')
plt.plot(epoch_list, loss_list_Rprop, color='black', label='Rprop')
plt.plot(epoch_list, loss_list_Adadelta, color='magenta', label='Adadelta')
plt.legend(loc='upper right')
plt.grid()
plt.show()    