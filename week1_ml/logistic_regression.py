import numpy as np


class LogisticRegression:
    def __init__(self, lr: float =0.01, epochs: int = 1000, lambda_reg: float = 0.01):
        self.lr = lr
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid, 注意数值稳定性"""
        z = np.clip(z, -500, 500)  #clip 的作用是截断
        return 1 / (1 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            linear = x @ self.weights + self.bias

            y_pred = self._sigmoid(linear)

            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

            #交叉熵
            ce_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            #正则化
            reg_loss = (self.lambda_reg / 2) * np.sum(self.weights ** 2)
            loss = ce_loss + reg_loss

            self.loss_history.append(loss)

            #计算梯度 
            dw = (1 / n_samples) * (x.T @ (y_pred - y)) + self.lambda_reg * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        返回概率值(0-1)
        """
        return self._sigmoid(x @ self.weights + self.bias)

    def predict(self, x:np.ndarray) -> np.ndarray:
        """返回类别 0 或 1"""
        proba = self.predict_proba(x)
        return (proba >= 0.5).astype(int)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """准确率"""
        return np.mean(self.predict(x) == y)


#===============测试代码===============#

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    x, y = make_classification(
        n_samples = 500,
        n_features = 2,
        n_redundant = 0,
        n_informative = 2,
        n_clusters_per_class = 1,
        random_state = 42
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    from data_processor import DataProcessor
    proc = DataProcessor(x_train, y_train)
    x_train_norm = proc.fit_transform(x_train) 
    x_test_norm = proc.transform(x_test)

    model = LogisticRegression(lr = 0.1, epochs = 1000, lambda_reg = 0.001)
    model.fit(x_train_norm, y_train)

    print(f"\n训练准确率:  {model.score(x_train_norm, y_train): .4f}")
    print(f"测试准确率： {model.score(x_test_norm, y_test): .4f}")

    from sklearn.linear_model import LogisticRegression as SKLR

    # sklearn 的逻辑回归（带 L2 正则，C 是正则化强度的倒数）
    sk_model = SKLR(C=1.0, max_iter=1000)
    sk_model.fit(x_train_norm, y_train)
    print(f"Sklearn 训练: {sk_model.score(x_train_norm, y_train):.4f}")
    print(f"Sklearn 测试: {sk_model.score(x_test_norm, y_test):.4f}")
    print(f"Sklearn 参数: w={sk_model.coef_}, b={sk_model.intercept_}")

