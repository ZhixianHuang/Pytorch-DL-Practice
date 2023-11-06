import torch
import matplotlib.pyplot as plt

# Model y_hat = w1*x^2 + w2*x + w3
# Model y_hat = 2*x^2 + 3*x + 4
# x = 4 y_hat = 48

x_data = [1.0, 2.0, 3.0]
y_data = [9.0, 18.0, 31.0]

w_1 = torch.tensor([1.0])
w_2 = torch.tensor([1.0])
b = torch.tensor([1.0])
w_1.requires_grad = True
w_2.requires_grad = True
b.requires_grad = True

def forward(x):
    return w_1*(x**2) + w_2*x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

epoch_list = []
cost_list = []
w_1.list =[]
w_2.list =[]
b.list =[]


print("Predict (before Training)", 4, forward(4).item())
for epoch in range(20000):
    sum = 0
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        
        sum += l.data
        
        print('\t grad', x_val, y_val, w_1.grad.item(), w_2.grad.item(), b.grad.item())
        w_1.data = w_1.data - 0.005 * w_1.grad.data
        w_2.data = w_2.data - 0.005 * w_2.grad.data
        b.data = b.data - 0.005 * b.grad.data
        
        w_1.grad.data.zero_()
        w_2.grad.data.zero_()
        b.grad.data.zero_()
        
    print("process ", epoch, l.item())
    w_1.list.append(w_1.data.item())
    w_2.list.append(w_2.data.item())
    b.list.append(b.data.item())
    epoch_list.append(epoch)
    cost_list.append(sum / 3)  # 原先的cost_list为单次的l.item()损失
                               # 将每个样本的损失累加到 sum 变量中，并在每个 epoch 结束后计算平均损失，然后将其添加到 cost_list 中。


print("Predict (after Training)", 4, forward(4).item())
print('w1=', w_1.data.item(), 'w2=', w_2.data.item(), 'b=', b.data.item(), )
# epoch = 100  w1= 2.0442910194396973 w2= 3.1519405841827393 b= 3.167264699935913
# epoch = 200  w1= 1.9192477464675903 w2= 3.405519962310791  b= 3.5119075775146484
# epoch = 500  w1= 1.8992730379104614 w2= 3.4410994052886963 b= 3.5814483165740967
# epoch = 1000 w1= 1.911397933959961  w2= 3.3785669803619385 b= 3.6595652103424072
# epoch = 5000 w1= 1.96751868724823   w2= 3.1387813091278076 b= 3.8751986026763916
# epoch =10000 w1= 1.9907352924346924 w2= 3.0395851135253906 b= 3.9644014835357666
# epoch =20000 w1= 1.9992514848709106 w2= 3.0032031536102295 b= 3.9971091747283936

# plt.plot(epoch_list, cost_list)  
plt.plot(epoch_list, w_1.list, color = 'blue')
plt.plot(epoch_list, w_2.list, color = 'green')
plt.plot(epoch_list, b.list, color = 'red')

plt.ylabel('cost')
plt.xlabel('epoch')
plt.grid()
plt.show()
    
        
    

