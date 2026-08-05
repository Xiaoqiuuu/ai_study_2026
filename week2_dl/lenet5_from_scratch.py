import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

#超参数
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#数据加载
transform = transforms.Compose([transforms.ToTensor(),  # -> (1, 28, 28), [0, 1]
                                transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准归一化
                                ])
# 使用脚本所在目录的绝对路径，避免相对路径在 Windows 上的权限问题
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
train_loader = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR, train = True, download = True, transform = transform), batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR, train = False, transform = transform), batch_size = BATCH_SIZE, shuffle = False)

#模型定义
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()


        #卷积块1: 1@28 * 28 -> 6@28 * 28 -> 6@14 * 14
        #参数量: 6 * (1 * 5 * 5 + 1) = 156
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 5, padding = 2)
        #池化: 2 * 2, stride = 2, 尺寸减半
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #卷积块2: 6@ 14 * 14 -> 16@ 10 * 10 -> 16@ 5 * 5 
        #尺寸计算: (14 - 5 + 0) / 1 + 1 = 10
        #参数量: 16 * (6 * 5 * 5 + 1) = 2416
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #全连接: 16@5 * 5= 400维向量 -> 120 ->84 -> 10
        self.fc1 = nn.Linear(16 * 5* 5, 120)    # 400 * 120 + 120 = 48120
        self.fc2 = nn.Linear(120, 84)    # 120 * 84 + 84 = 10164
        self.fc3 = nn.Linear(84, 10)    # 84 * 10 + 10 = 850

    def forward(self, x): 
        #卷积块1
        x = self.conv1(x)   #(N, 1, 28, 28) -> (N, 6, 28, 28)
        x = torch.relu(x)
        x = self.pool1(x)    # -> (N, 6, 14, 14) 

        #卷积块2 
        x = self.conv2(x)   # -> (N, 16, 10, 10)
        x = torch.relu(x)
        x = self.pool2(x)    # -> (N, 16, 5, 5)

        #展平：把(N, 16, 5, 5)展平成 (N, 400)
        #x.size(0) 是 batch_size, -1自动计算 16 * 5 * 5 = 400
        x = x.view(x.size(0), -1)

        #全连接
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)     #输出logits, 不经过SoftMax(CrossEntropyLoss内部会做)
        return x

model = LeNet5().to(DEVICE)

print("=" * 40)
print("逐层参数量分析")
total = 0
for name, p in model.named_parameters():
    n = p.numel()
    total += n
    print(f" {name:25s}: {n:>7,}")
print(f" {'Total':25s}: {total:>7,}")
print("=" * 40)


#==========损失函数 & 优化器 =========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)

#========== 训练 & 测试循环 =========
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim = 1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx:3d}, Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total

def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim = 1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), correct / total

#===================主循环====================
print("开始训练 LeNet-5...")
for epoch in range (EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = test_epoch(model, test_loader, criterion, DEVICE)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"  Train_loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Test_lose: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("-" * 40)

print("训练完成！")
