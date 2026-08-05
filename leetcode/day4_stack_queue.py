#20 155 232
import numpy as np
class Solution_20:
    def isValid(self, s: str) -> bool:
        qua = []
        dic = {')': '(', ']': '[', '}': '{'}
        for ele in s:
            if ele == "(" or ele == "[" or ele == "{":
                qua.append(ele)

            if ele in dic:
                if qua == [] or qua[-1] != dic[ele]:
                    return False
                else: qua.pop()
        if qua == []:
            return True
        return False


class MinStack:
    def __init__(self):
        self.stack = []   # 主栈
        self.min = []     # 辅助栈（单调非递增）

    def push(self, value: int) -> None:
        self.stack.append(value)
        # 改为 <=，确保重复最小值也入栈
        if not self.min or value <= self.min[-1]:
            self.min.append(value)

    def pop(self) -> None:
        if not self.stack:
            return
        # 先取出栈顶元素，再弹出主栈
        ele = self.stack[-1]
        self.stack.pop()   # 必须执行这一行！
        
        # 如果弹出的元素是当前最小值，辅助栈也要弹出
        if self.min and ele == self.min[-1]:
            self.min.pop()

    def top(self) -> int:
        """获取栈顶元素（不弹出）"""
        if not self.stack:
            raise IndexError("栈为空")
        return self.stack[-1]

    def getMin(self) -> int:
        """获取当前最小值"""
        if not self.min:
            raise IndexError("栈为空")
        return self.min[-1]



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(value)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


class MyQueue:

    def __init__(self):
        self.front = []
        self.back = []
        self.size = 0
        
    def push(self, x: int) -> None:
        if len(self.front) == 0:
            self.front.append(x)
        else:    
            self.back.append(x)
        self.size += 1
    def pop(self) -> int:
        x = self.front.pop()
        if len(self.front) == 0:
            while len(self.back) != 0:
                self.front.append(self.back.pop())
        self.size -= 1
        return x
    def peek(self) -> int:
        if self.size == 0:
            raise Exception("队列长为0， 操作失败")
        return self.front[-1]

    def empty(self) -> bool:
        if self.size == 0:
            return True
        return False


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()