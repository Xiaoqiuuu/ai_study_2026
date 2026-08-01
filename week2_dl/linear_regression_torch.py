import torch
import numpy as np

class LinearRegressionTorch:
    """
    用 Pytorch重写 Day1 的线性回归
    核心差异: Pytorch 自动计算梯度, 不需要手动计算dw, db
    """
    def __init__(self, lr: float = 0.1, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs

        #pytorch用 nn.Parameter 包装可训练参数
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, x: torch.tensor, y: torch.tensor):
        n_samples, n_features = x.shape

        #初始化参数
        self.w = torch.zeros(n_features, 1, requires_grad = True)
        self.b = torch.zeros(1, requires_grad = True)

        for epoch in range(self.epochs):
            #前向传播
            y_pred = x @ self.w + self.b

            #计算损失(MSE)
            loss = torch.mean((y_pred - y.view(-1, 1)) ** 2)
            self.loss_history.append(loss.item())

            #反向传播
            loss.backward()

            #更新参数(用torch.no_grad(), 否则会被追踪)
            with torch.no_grad():
                self.w -= self.lr * self.w.grad
                self.b -= self.lr * self.b.grad

                #必须手动清空梯度，否则梯度会累加
                self.w.grad.zero_()
                self.b.grad.zero_()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, x:torch.tensor):
        return x @ self.w + self.b


#================测试==================
if __name__ == "__main__":
    #生成数据
    torch.manual_seed(42)
    x = torch.rand(100, 1)
    y = 3 * x.squeeze() + 4 + torch.randn(100) * 0.1

    #训练
    model = LinearRegressionTorch(lr = 0.1, epochs = 1000)
    model.fit(x,y)

    print(f"\n学习参数: w = {model .w.item():.4f}, b = {model.b.item():.4f}")
    print(f"真是参数: w = 3.0000, b = 4.0000")

    #对比: 用Pytorch的nn.module实现
    print("\n=== Pytorch的官方实现 ===")
    import torch.nn as nn

    #定义模型
    official_model = nn.Linear(in_features = 1, out_features = 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(official_model.parameters(), lr = 0.1)

    for epoch in range(1000):
        y_pred = official_model(x).squeeze()
        loss = criterion(y_pred, y)

        optimizer.zero_grad() #清零梯度
        loss.backward() #反向传播
        optimizer.step() #更新参数

    for name, param in official_model.named_parameters():
        print(f"{name}: {param.data}")
