# 509, 70, 50
class Solution_509:
    def fib(self, n: int) -> int:
        if n == 0: return 0
        if n == 1: return 1
        return self.fib(n-1) + self.fib(n-2)

class Solution_50:
    def myPow(self, x: float, n: int) -> float:
        # 1. 处理负数次幂
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1.0
        base = x
        exponent = n
        
        while exponent > 0:
            # 检查当前二进制位是不是 1 (exponent & 1)
            if exponent & 1:
                result *= base
            
            # 底数平方，准备处理下一位
            base *= base
            # 指数右移一位 (相当于除以2)
            exponent >>= 1
        
        return result


class Solution_70:
    def climbStairs(self, n: int) -> int:
        steps = [1, 2]
        stair = 2
        while stair < n:
            steps.append(steps[-2] + steps[-1])
            stair += 1
        if n == 1 or n == 2: return steps[n-1]
        return steps[-1]