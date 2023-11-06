import torch

criterion = torch.nn.CrossEntropyLoss()
# 输出需要是一个长整型的张量
y_label = torch.LongTensor([2, 0, 1]) 
y_pred_1 = torch.Tensor([[0.1, 0.2, 0.9],  # 2
                       [1.1, 0.1, 0.2],    # 0
                       [0.2, 2.1, 0.1]])   # 1  小loss
y_pred_2 = torch.Tensor([[0.8, 0.2, 0.3],  # 0
                         [0.2, 0.3, 0.5],  # 2
                         [0.2, 0.2, 0.5]]) # 2  大loss

loss_1 = criterion(y_pred_1, y_label)
loss_2 = criterion(y_pred_2, y_label)

print('loss1= {:.4f}'.format(loss_1))
print('loss2= {:.4f}'.format(loss_2))
