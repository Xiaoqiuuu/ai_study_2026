# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


import torch
import numpy as np
# =========== 创建tensor =============
#从列表创建
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

#随机初始化
w = torch.rand(3, 2)    #标准正态分布
w = torch.zeros(3, 2)   #全0   
w = torch.ones(3, 2)    #全1

#类似numpy的操作
print(w.shape) 
print(w.T)
print(w @ w.T)

#GPU张量 
if torch.cuda.is_available():
    w_gpu = w.cuda()    #搬到gpu
    w_gpu = torch.rand(3, 2, device = 'cuda') #直接在gpu上创建

#===============Autograd================
x = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)     #requires_grad = True表示追踪这个张量的所有运算，用于反向求导

y = (x ** 2).sum()
print(y) 

#反向传播
y.backward()

#查看梯度
print(x.grad)