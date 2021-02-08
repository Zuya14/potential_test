import torch

def func0(x, y):
    return torch.norm((x - 2.1)**2 + (y+7.9)**2)

eta = 0.1
x = torch.randn(1, requires_grad=True)
y = torch.randn(1, requires_grad=True)
print(x, y)

for _ in range(100):
    z = func0(x, y)
    z.backward()
    with torch.no_grad(): 
        x = x - eta * x.grad
        y = y - eta * y.grad
    print(x, y)
    x.requires_grad = True 
    y.requires_grad = True 