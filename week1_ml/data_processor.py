import numpy as np
from typing import Tuple, Optional


class DataProcessor:
    """
    数据处理类, 支持加载, 清洗, 划分数据集
    """
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.x = x.astype(np.float64)
        self.y = y.astype(np.float64) if y is not None else None

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    @classmethod
    def from_file(cls, filepath: str, delimiter: str = ","):
        """从文件创建实例"""
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1)
        x = data[:, :-1]
        y = data[:, -1]
        return cls(x, y)  # 调用 __init__

    def normalize(self) -> "DataProcessor":
        """
        Z-score标准化: (x - mean) / std
        返回 self 支持链式调用
        """
        self.x_mean = np.mean(self.x, axis = 0)
        self.x_std = np.std(self.x, axis =0)
        #防止除以0
        self.x_std[self.x_std == 0] = 1.0
        self.x = (self.x - self.x_mean) / self.x_std
        return self

    def split(self, test_size: float = 0.2, seed: int = 886) -> Tuple:
        """
        划分训练集和测试集
        test_size: 测试集占比
        seed: 随机种子
        return: (x_train, y_train, x_test, y_test)
        """
        np.random.seed(seed)
        n = len(self.X)
        idx = np.random.permutation(n)
        split = int(n * (1 - test_size))
        
        train_idx, test_idx = idx[:split], idx[split:]
        
        if self.y is not None:
            return (self.x[train_idx], self.x[test_idx],
                    self.y[train_idx], self.y[test_idx])
        return self.x[train_idx], self.x[test_idx]

    def fit(self, x: np.ndarray):
        """
        在训练集上计算均值和标准差
        只调用一次！
        """
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1.0  # 防止除零
        return self
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        用训练集的参数转换任意数据（训练集或测试集）
        """
        if self.mean is None:
            raise ValueError("必须先调用 fit()")
        return (x - self.mean) / self.std
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """fit + transform 的快捷方式"""
        self.fit(x)
        return self.transform(x)
    
    def add_polynomial_features(self, x: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        添加多项式特征
        degree: 多项式的最高次数
        返回 self 支持链式调用
        """
        x_poly = x.copy()
        x_original = x.copy()
        for d in range(2, degree + 1):
            x_poly = np.concatenate([x_poly, x_original ** d], axis=1)
        return x_poly
    
    def add_polynomial_features(self, degree: int = 2) -> "DataProcessor":
        """
        
        """
        x_original = self.x.copy()  
        x_poly = self.x.copy()
        for d in range(2, degree + 1):
            x_poly = np.concatenate([x_poly, x_original ** d], axis=1)
        self.x = x_poly
        return self

    #================测试代码===========#
if __name__ == "__main__":
    #生成数据
    np.random.seed(432)
    x = np.random.rand(200, 1)
    y = 3 * x.squeeze() + 4 + np.random.randn(200) * 0.5

    processor = DataProcessor(x, y)
    x_train, x_test, y_train, y_test = processor.normalize().split(test_size=0.2)

    print("训练集大小:", x_train.shape, y_train.shape)
    print("测试集大小:", x_test.shape, y_test.shape)
    print(f"特征均值：{np.mean(x_train, axis = 0 )}")
    from linear_regression import LinearRegression
    model = LinearRegression(learning_rate=0.05, epochs=1000)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    print(f"训练集R^2分数: {train_score}")
    test_score = model.score(x_test, y_test)
    print(f"测试集R^2分数: {test_score}")
    print(f"模型参数: 权重={model.weights}, 偏置={model.bias}")

    from matplotlib import pyplot as plt
    plt.scatter(x_test, y_test, color='blue', label='Test Data')
    plt.plot(x_test, model.predict(x_test), color='red', label='Prediction')
    plt.title("Linear Regression Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # #文件路径
    # filepath = "data.csv"
    # processor = DataProcessor.from_file(filepath)
    # processor.normalize().add_polynomial_features(degree=3)
    # processor.split(test_size=0.2)
    # model = LinearRegression(learning_rate=0.01, epochs=1000)
    # model.fit(processor.x, processor.y)
    # train_score = model.score(processor.x, processor.y)
    # print(f"训练集R^2分数: {train_score}")
    # test_score = model.score(processor.x, processor.y)
    # print(f"测试集R^2分数: {test_score}")
    # print(f"模型参数: 权重={model.weights}, 偏置={model.bias}")
    # from matplotlib import pyplot as plt
    # plt.scatter(processor.x[:, 0], processor.y, color='blue', label='Data')
    # plt.plot(processor.x[:, 0], model.predict(processor.x), color='red', label='Prediction')
    # plt.title("Polynomial Regression Prediction")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.show()
