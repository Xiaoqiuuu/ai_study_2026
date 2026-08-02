import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class NeuralNetwork(nn.Module):
    """
    全连接网络: 输入(784) -> 隐藏层(256) -> 输出(10)
    784 = 28 * 28
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # nn.Linear 底层就是 y = xW^T + b, w 和 b 自动初始化
        self.layer1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)    #dropout函数，随即丢弃神经元，防止过拟合
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        """
        前向传播
        forward定义计算图, backward 自动沿着计算图反向求导
        """
        x = x.view(-1, 28 * 28)
        x = self.layer1(x)    #线性变换
        x = self.relu(x)      #非线性激活函数，否则多层网络退化成单层
        x = self.dropout(x)   #只在训练时生效，eval时自动关闭
        x = self.layer2(x)    #输出十个类别的logits，表示每个类别的预测得分

        return x


#=================训练流程==================
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()    #将模型切换到训练模式，开启dropout，BatchBorm 用当前batch的均值和方差做标准化，并更新全局统计量

    for batch_idx, (data, target) in enumerate(train_loader):
        #train_loader 把数据集分成很多batch
        #data 是当前batch的特征，形如(64, 1, 28, 28)
        #target 是当前batch的标签， 形如(64, )
        #batch_idx 是第几批

        data, target = data.to(device), target.to(device)
        # Pytorch 要求运算的两个张量再同一设备

        optimizer.zero_grad()
        output = model(data)    # 前向传播
        loss = criterion(output, target)
        #criterion 是nn.CrossEntropyLoss(), 内部做了两件事: LogSoftmax(把 logits 变成概率对数) + NLLLoss(负对数似然)
        #输出一个标量张量(tensor(2.3456))
        #CrossEntropyLoss 是分类任务的标准选择，内部集成了softmax, 数值稳定性更好
        loss.backward()    #反向传播
        optimizer.step()    #更新参数

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch}[{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss:{loss.item():.4f}')



def test(model, device, test_loader):
    model.eval()    #model切换到eval模式，关闭dropout，固定Batchnorm
    test_loss = 0
    correct = 0

    with torch.no_grad(): #测试时不计算梯度
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 预测: output 是 (batch，10)的 logits，取最大值的索引就是类别
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        return accuracy



#================主程序================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    #数据预处理：转成Tensor并归一化到(0, 1)
    transform = transforms.Compose([transforms.ToTensor(),    #将PIL Image转为Tensor，并除以255
                                    transforms.Normalize((0.1307,),(0.3081,))   #MNIST的均值和标准差  
                                    ])

    #下载数据
    train_dataset = datasets.MNIST('data', train = True, download = True, transform = transform)
    test_dataset = datasets.MNIST('data', train = False, transform = transform)

    #DataLoader: 批量加载，自动打乱，多进程加速
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

    #模型，损失函数，优化器
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9) #momentum: 加速收敛，抑制震荡


    #训练
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader)

    #保存模型
    torch.save(model.state_dict(), "mnist_model.pth")
    print("模型已保存")