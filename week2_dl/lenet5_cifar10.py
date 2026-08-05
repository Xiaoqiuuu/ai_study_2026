import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.4914, 0.4822, 0.4465),
                                    std=(0.2023, 0.1994, 0.2010) 
                                    )
                                ])

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(DATA_DIR, train =True, download = True, transform = transform), batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(DATA_DIR, train = False, transform = transform), batch_size = BATCH_SIZE, shuffle = False)



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(16 *6 *6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        return x


model = LeNet5().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)

def train_epochs(model, loader, criterion, optimizer, device):
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
            print(f"Batch: {batch_idx}, Loss: {loss.item():.4f}")

    return  total_loss / len(loader), correct / total

def test_epochs(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim = 1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return total_loss / len(loader), correct / total

print("开始训练LeNet5 ...")
for epoch in range(EPOCHS):
    train_loss, train_accuracy = train_epochs(model, train_loader, criterion, optimizer = optimizer, device = DEVICE)
    test_loss, test_accuracy = test_epochs(model, test_loader, criterion, device = DEVICE)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"  Train_loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
    print(f"  Test_lose: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    print("-" * 40)
    
print("训练完成！")





