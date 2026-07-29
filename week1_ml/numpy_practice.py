import numpy as np

a = np.array([1, 2, 3])  
b = np.array([[1,2], [3,4], [5,6]])

print(a)  #输出[1, 2, 3]
print(b)  #输出[[1, 2]
          #     [3, 4]
          #     [5, 6]]

print(b.shape)  #输出(3, 2)
print(b.reshape(2, 3))  #输出[[1, 2, 3]
                        #     [4, 5, 6]]
                        #reshape()函数可以改变数组的形状，但不改变数据本身。
print(b.T)  #输出[[1, 3, 5]
            #     [2, 4, 6]]

c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])

print(c + d)  #输出[[ 6  8]
              #     [10 12]]
print(c * d)  #输出[[ 5 12]
              #     [21 32]]
              # * 运算符执行元素级乘法
print(c @ d)  #输出[[19 22]
              #     [43 50]]
              # @ 运算符执行矩阵乘法
print(np.dot(c, d))  #输出[[19 22]
                    #     [43 50]]
                    #dot()函数也执行矩阵乘法

# 广播机制
e = np.array([[1], [2], [3]])
f = np.array([10, 20])
print(e + f)  #输出[[11 21]
              #     [12 22]
              #     [13 23]]
              #广播机制允许不同形状的数组进行运算，NumPy会自动扩展较小的数组以匹配较大数组的形状。

print(np.mean(a))  #输出2.0
print(np.sum(b, axis=0))  #输出[9 12]
print(np.argmax(a))  #输出2
                     #argmax()函数返回数组中最大值的索引。

def Gaussian_elimination(A, b):
    """使用高斯消元法求解线性方程组 Ax = b"""
    n = len(b)
    # 将增广矩阵 [A|b] 进行行变换
    for i in range(n):
        # 寻找主元
        max_row = np.argmax(np.abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        
        # 消元
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # 回代求解
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    
    return x
def inverse(matrix):
    """计算矩阵的逆"""
    m = matrix.shape[0]
    n = matrix.shape[1] 
    if m != n:
          raise ValueError("矩阵必须是方阵才能求逆")

    # 创建增广矩阵 [A|I]
    augmented_matrix = np.hstack((matrix, np.eye(m)))
    # 对增广矩阵进行行变换，将其转换为单位矩阵
    for i in range(m):
        # 寻找主元
        max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
        
        # 消元
        for j in range(i + 1, m):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
    
    # 回代求解
    for i in range(m - 1, -1, -1):
        augmented_matrix[i, :] /= augmented_matrix[i, i]
        for j in range(i):
            factor = augmented_matrix[j, i]
            augmented_matrix[j, :] -= factor * augmented_matrix[i, :]
    
    # 提取逆矩阵
    inverse_matrix = augmented_matrix[:, m:]
    return inverse_matrix

m = np.array([[4, 7], [2, 6]])
print(inverse(m))  #输出[[ 0.6 -0.7]
                   #     [-0.2  0.4]]

print(m @ inverse(m))  #输出[[1. 0.]
                       #     [0. 1.]]

                       