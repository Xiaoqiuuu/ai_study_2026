import numpy as np


class LinearRegression:
    """
    线性回归模型类 使用梯度下降训练
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None 
        self.loss_history = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        训练模型
        x: shape (n_samples, n_features)
        y: shape (n_samples,)
        """
        n_samples , n_features = x.shape

        #1. 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        #2. 梯度下降
        for epoch in range(self.epochs):
            #向前传播 计算预测值 y_pred = x @ weights + bias
            y_pred = self.predict(x)

            #计算损失函数 MSE = 1/n * sum((y - y_pred)^2)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            #计算梯度
            dw = (-2 / n_samples) * x.T @ (y - y_pred)
            db = (-2 / n_samples) * np.sum(y - y_pred)

            #更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测函数
        x: shape (n_samples, n_features)
        return: shape (n_samples,)
        """
        return x @ self.weights + self.bias

    def score(self, x: np.ndarray, y: np.array) -> float:
        """
        计算模型的R^2分数
        R^2 = 1 - (sum((y - y_pred)^2) / sum((y - mean(y))^2))
        """
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


#============测试代码===========#

if __name__ == "__main__":
    #生成假数据： y = 3x + 4 + 噪声
    np.random.seed(432)
    x = np.random.rand(100, 1)
    y = 3 * x.squeeze() + 4 + np.random.randn(100) * 0.1

    #划分训练集和测试集
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    #训练模型
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(x_train, y_train)

    #评估
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"\n训练集 R^2: {train_score:.4f}")
    print(f"测试集 R^2: {test_score:.4f}")
    print(f"真实权重: 3, 偏置: 4")
    print(f"预测权重: {model.weights[0]:.4f}, 偏置: {model.bias:.4f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, color='blue', label='True')
    plt.plot(x_test, model.predict(x_test), color='red', label='Predicted')
    plt.title("Linear Regression Fit")
    plt.legend()

    plt.savefig("linear_regression_result(lr=0.01).png")
    plt.show()