import torch

a = [1, 2, 3]
b = [4, 5, 6]
a = torch.tensor(a).reshape(1,3)
b = torch.tensor(b).reshape(1,3)
outputs = [a, b]
print(torch.cat(outputs, dim=1))