import torch

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])

z = x.add(y)
print(x, z)

a = {1: 1, 2: 2}
print(list(a.keys()))