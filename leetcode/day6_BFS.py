#102, 101, 111
# Definition for a binary tree node.

from typing import Optional
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque
class Solution_102:
    def levelOrder(self, root: Optional[TreeNode]) -> list[list[int]]:
        visit = deque()
        output = []
        if root is None: return []
        visit.append(root)

        while visit:
            val = []
            n = len(visit)

            for _ in range(n):
                node = visit.popleft()
                val.append(node.val)
                if node.left: visit.append(node.left)
                if node.right: visit.append(node.right)
            output.append(val)
        return output

class Solution_101:
    def isSymmetric_recrusion(self, root: Optional[TreeNode]) -> bool:
        if root is None: return True

        def check(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
            # 1. 都为空，对称
            if not left and not right:
                return True
            # 2. 其中一个为空，或值不等，不对称
            if not left or not right or left.val != right.val:
                return False
            
            # 3. 递归检查：左的左 vs 右的右；左的右 vs 右的左
            return check(left.left, right.right) and check(left.right, right.left)
        
        return check(root.left, root.right)

    def isSymmetric_iteration(self, root: Optional[TreeNode]) -> bool:
        if root is None: return True
        q = deque()
        q.append(root.left)
        q.append(root.right)

        while q:
            a = q.popleft()
            b = q.popleft()

            if not a and not b: continue
            if a is None or b is None or a.val != b.val:
                return False
            q.append(a.left)
            q.append(b.right)
            q.append(a.right)
            q.append(b.left)

        return True


class Solution_111:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if root is None: return 0
        dep = 1
        q = deque([root])
        while q:
            size = len(q)
            for _ in range(size):
                node = q.popleft()
                if not node.left and not node.right:
                    return dep
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)

            dep += 1
                
        
        
