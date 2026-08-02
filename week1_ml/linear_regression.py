"""
线性回归（Linear Regression）的手动实现
========================================
使用梯度下降法（Gradient Descent）求解最小二乘问题。
模型假设: y = X @ w + b，即目标值是输入特征的线性组合。

核心思想:
  1. 定义损失函数（MSE: 均方误差）
  2. 计算损失函数对各参数的梯度
  3. 沿梯度反方向更新参数，使损失逐渐减小
  4. 重复迭代直到收敛

数学公式:
  - 预测值:  ŷ = Xw + b
  - 损失函数: J = (1/n) * Σ(y - ŷ)²
  - 权重梯度: ∂J/∂w = (-2/n) * Xᵀ(y - ŷ)
  - 偏置梯度: ∂J/∂b = (-2/n) * Σ(y - ŷ)
"""

import numpy as np


class LinearRegression:
    """
    线性回归模型类，使用梯度下降法训练。

    这是一个简单但完整的实现，展示了机器学习模型的三个核心方法:
      - fit():   用训练数据拟合模型
      - predict(): 用训练好的模型做预测
      - score():  评估模型的拟合优度 (R²)

    使用示例:
      >>> model = LinearRegression(learning_rate=0.01, epochs=1000)
      >>> model.fit(X_train, y_train)
      >>> y_pred = model.predict(X_test)
      >>> r2 = model.score(X_test, y_test)
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        """
        初始化线性回归模型。

        参数:
            learning_rate: 学习率（步长），控制每次参数更新的幅度。
                          太大可能导致震荡/不收敛，太小则收敛过慢。
            epochs: 迭代轮数，完整遍历训练数据的次数。
        """
        self.lr = learning_rate           # 学习率，梯度下降的步长
        self.epochs = epochs               # 最大迭代次数
        self.weights = None                # 权重向量 w，形状 (n_features,)
        self.bias = None                   # 偏置项 b，一个标量
        self.loss_history = []             # 记录每次迭代的损失值，用于画损失曲线

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        训练模型 — 使用批量梯度下降（Batch Gradient Descent）拟合数据。

        算法流程:
          1. 将权重 w 和偏置 b 初始化为零
          2. 每轮迭代中:
             a. 前向传播: 用当前参数计算所有样本的预测值 ŷ = Xw + b
             b. 计算损失: MSE = (1/n) * Σ(y - ŷ)²
             c. 反向传播: 求出损失关于 w 和 b 的梯度
             d. 参数更新: w = w - lr * dw,  b = b - lr * db

        参数:
            x: 训练特征矩阵，形状为 (样本数, 特征数)
            y: 训练标签向量，形状为 (样本数,)
        """
        n_samples, n_features = x.shape  # 获取样本数和特征数

        # ===== 1. 初始化参数 =====
        # 权重初始化为全零向量（简单场景下足够用）
        self.weights = np.zeros(n_features)
        # 偏置初始化为 0
        self.bias = 0

        # ===== 2. 梯度下降主循环 =====
        for epoch in range(self.epochs):
            # --- 2a. 前向传播: 计算预测值 ---
            # y_pred = X @ w + b，矩阵乘法 @ 等价于 X.dot(w)
            y_pred = self.predict(x)

            # --- 2b. 计算损失 (MSE: Mean Squared Error) ---
            # MSE = (1/n) * Σ(y_i - ŷ_i)²
            # 损失越小，说明预测值与真实值越接近
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)  # 记录下来便于后续可视化

            # --- 2c. 计算梯度 ---
            # 误差项 e = (y - ŷ)，形状 (n_samples,)
            # dw = ∂J/∂w = (-2/n) * Xᵀ @ e
            #   Xᵀ 形状 (n_features, n_samples)，e 形状 (n_samples,)
            #   结果形状 (n_features,)，与 w 一致
            dw = (-2 / n_samples) * x.T @ (y - y_pred)

            # db = ∂J/∂b = (-2/n) * Σ e_i
            #   所有样本误差之和再缩放
            db = (-2 / n_samples) * np.sum(y - y_pred)

            # --- 2d. 更新参数（梯度下降核心步骤）---
            # 沿梯度的反方向更新: w_new = w_old - lr * dw
            # 因为梯度指向损失上升最快的方向，所以反方向就是下降最快的方向
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # 每 100 轮打印一次当前损失，方便观察训练进度
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测函数 — 用训练好的参数计算预测值。

        计算公式: ŷ = X @ w + b
          - X @ w 计算每个样本的加权特征和
          - 加上偏置 b 得到最终预测值

        参数:
            x: 输入特征矩阵，形状 (样本数, 特征数)
        返回:
            y_pred: 预测值向量，形状 (样本数,)
        """
        return x @ self.weights + self.bias

    def score(self, x: np.ndarray, y: np.array) -> float:
        """
        计算模型的 R² 分数（决定系数，Coefficient of Determination）。

        R² 衡量模型拟合优度，取值范围通常为 [0, 1]:
          - R² = 1: 完美预测（残差为 0）
          - R² = 0: 模型等价于用均值预测（没有学到任何规律）
          - R² < 0: 模型还不如直接用均值预测

        计算公式:
          R² = 1 - SS_res / SS_tot
          其中:
            SS_res = Σ(y_i - ŷ_i)²     — 残差平方和（模型没解释的部分）
            SS_tot = Σ(y_i - ȳ)²       — 总平方和（数据的总变异）

        直观理解: R² 表示"目标值的变化中有百分之多少能被模型解释"。

        参数:
            x: 特征矩阵, shape (n_samples, n_features)
            y: 真实标签, shape (n_samples,)
        返回:
            r2_score: R² 分数 (float)
        """
        y_pred = self.predict(x)                       # 用当前参数做预测
        ss_res = np.sum((y - y_pred) ** 2)             # 残差平方和（模型没解释的变异）
        ss_tot = np.sum((y - np.mean(y)) ** 2)         # 总平方和（数据的总变异）
        return 1 - (ss_res / ss_tot)                   # R² = 1 - 未解释/总


# =============================================#
#                   测试代码                      #
# =============================================#

if __name__ == "__main__":
    # ===== 生成模拟数据集 =====
    # 真实关系: y = 3x + 4 + 噪声
    # 我们的目标是让模型从数据中"学出" w≈3, b≈4
    np.random.seed(432)  # 固定随机种子，保证每次运行结果一致
    x = np.random.rand(100, 1)  # 100 个样本，1 个特征，取值范围 [0, 1)
    # 生成标签: y_true = 3*x + 4，加上均值为 0、标准差为 0.1 的高斯噪声
    y = 3 * x.squeeze() + 4 + np.random.randn(100) * 0.1

    # ===== 划分训练集 (80%) 和测试集 (20%) =====
    # 训练集: 用来拟合模型参数
    # 测试集: 用来评估模型的泛化能力（训练中从未见过）
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ===== 训练模型 =====
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(x_train, y_train)

    # ===== 评估模型 =====
    # 分别计算训练集和测试集上的 R² 分数
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"\n训练集 R²: {train_score:.4f}")
    print(f"测试集 R²: {test_score:.4f}")
    # 对比学到的参数与真实参数
    print(f"真实权重: 3, 偏置: 4")
    print(f"预测权重: {model.weights[0]:.4f}, 偏置: {model.bias:.4f}")

    # ===== 可视化 =====
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    # 左图: 损失曲线 — 观察损失是否收敛（如果一直不降说明学习率可能太大）
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    # 右图: 拟合效果 — 蓝色散点为真实数据，红色直线为模型预测
    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, color='blue', label='True')       # 测试集真实值
    plt.plot(x_test, model.predict(x_test), color='red', label='Predicted')  # 模型拟合的直线
    plt.title("Linear Regression Fit")
    plt.legend()

    plt.savefig("linear_regression_result(lr=0.01).png")  # 保存图片到磁盘
    plt.show()  # 显示图片