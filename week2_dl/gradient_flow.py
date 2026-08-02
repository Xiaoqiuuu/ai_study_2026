# 简化版：手动追踪一个2层网络的梯度
import torch

x = torch.tensor([[1.0, 2.0]], requires_grad=True)  # 输入(1,2)
w1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)  # (2,2)
b1 = torch.tensor([0.0, 0.0], requires_grad=True)

# 第一层
z1 = x @ w1 + b1        # z1 = [1.5, 1.5]
a1 = torch.relu(z1)     # a1 = [1.5, 1.5]

w2 = torch.tensor([[1.0], [1.0]], requires_grad=True)  # (2,1)
b2 = torch.tensor([0.0], requires_grad=True)

# 第二层
z2 = a1 @ w2 + b2       # z2 = [3.0]
loss = (z2 - 5.0) ** 2  # MSE

loss.backward()

print("w2 grad:", w2.grad)  # 应该是多少？手动算一下验证
print("w1 grad:", w1.grad)  # 链式法则传了两层，应该是多少

import torch
import torch.nn as nn

# 构造一个深网络（10层），观察梯度大小
class DeepNet(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(10):
            self.layers.append(nn.Linear(100, 100))
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        self.output = nn.Linear(100, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output(x)

# 训练一步，记录每层梯度的范数
for act in ['sigmoid', 'relu']:
    model = DeepNet(activation=act)
    x = torch.randn(32, 100)
    y = torch.randn(32, 1)
    
    criterion = nn.MSELoss()
    loss = criterion(model(x), y)
    loss.backward()
    
    print(f"\n=== {act.upper()} ===")
    for i, layer in enumerate(model.layers):
        grad_norm = layer.weight.grad.norm().item()
        print(f"Layer {i}: grad_norm = {grad_norm:.6f}")