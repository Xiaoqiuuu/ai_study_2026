# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import Optional
class Solution_104:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:return 0
        if root.left == None and root.right == None:
            return 1
        elif root.right == None:
            return self.maxDepth(root.left) + 1
        elif root.left == None:
            return self.maxDepth(root.right) + 1
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

class Solution_226:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None: return 
        else:
            self.invertTree(root.left)
            self.invertTree(root.right)
            root.left, root.right = root.right, root.left
        return root

class Solution_112:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root == None:return False
        if root.left is None and root.right is None:return root.val == targetSum
        return (self.hasPathSum(root.left,targetSum - root.val) or self.hasPathSum(root.right,targetSum - root.val))
        