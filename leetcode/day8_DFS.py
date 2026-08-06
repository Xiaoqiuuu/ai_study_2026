# 94 543 124 230
from typing import Optional
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution_94:
    def inorderTraversal(self, root: Optional[TreeNode]) -> list[int]:
        res = []
        
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            res.append(node.val)
            dfs(node.right)
        
        dfs(root)
        return res

    def inorderTraversal_iteration(self, root: Optional[TreeNode]) -> list[int]:
        res = []
        stack = []
        cur = root
        
        while cur or stack:
            # 阶段1：一直向左走，沿途节点压栈
            while cur:
                stack.append(cur)
                cur = cur.left
            
            # 阶段2：左边走到底了，弹出栈顶（也就是最左的节点）
            cur = stack.pop()
            res.append(cur.val)
            
            # 阶段3：转向右子树，继续重复上述过程
            cur = cur.right
        
        return res



class Solution_543:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        diameter = 0
        
        def depth(node: Optional[TreeNode]) -> int:
            nonlocal diameter
            if not node:
                return 0
            left = depth(node.left)
            right = depth(node.right)
            # 更新直径：经过当前节点的最长路径（左右子树高度之和）
            diameter = max(diameter, left + right)
            # 返回当前节点的高度（边数）
            return max(left, right) + 1
        
        depth(root)
        return diameter


class Solution_124:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        maxPathSum = 0
                
        def sum(node: Optional[TreeNode]) -> int:
            nonlocal maxPathSum
            if not node:
                return 0
            if not node.left and not node.right: return 0 if node.val < 0 else node.val
            else:
                left = sum(node.left)
                right = sum(node.right)
                # 更新直径：经过当前节点的最长路径（左右子树高度之和）
                diameter = max(diameter, left + right)
                # 返回当前节点的高度（边数）
                return max(left, right) + node.val
                
        sum(root)
        return maxPathSum

class Solution_230:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.count = 0
        self.result = None
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            self.count += 1
            if self.count == k:
                self.result = node.val
                return
            inorder(node.right)
        
        inorder(root)
        return self.result