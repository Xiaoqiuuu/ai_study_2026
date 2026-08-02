# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution_206_iteration:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head
        while cur:
            next_node = cur.next
            cur.next = prev
            prev = cur
            cur = next_node
        head = prev
        return head
class Solution_206_recrusion:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def _reverse_recursive(self, node):
        """递归内部函数：返回反转后的新头节点"""
        # 递归终止条件：空节点 或 到达尾节点
        if node is None or node.next is None:
            return node

        # 递归反转后续子链表，new_head 永远是原链表的尾节点（反转后的头）
        new_head = self._reverse_recursive(node.next)

        # 回溯阶段：让当前节点的下一个节点（原后继）指向自己
        # 例如当前是 1，node.next 是 2，执行 2.next = 1
        node.next.next = node
        # 断开当前节点原来的 next，防止成环（1.next 先置为 None）
        node.next = None

        # 将反转后的头节点向上传递
        return new_head

    def reverseList(self, head: ListNode) -> ListNode:
        """递归法反转链表（对外调用接口）"""
        if self == None:
            return
        cur = self._reverse_recursive(head)
        return cur

class Solution_21:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy
        while list1 and list2:
            if list1.val <= list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        tail.next = list1 or list2   # 剩余部分直接接上，安全（即使 tail 是 dummy 也行）
        return dummy.next
class Solution_141:
    def hasCycle(self, head):
        """
        哈希集合：遍历并记录已访问节点，若再次遇到同一节点则说明有环。
        时间复杂度: O(n)  空间复杂度: O(n)
        :type head: ListNode
        :rtype: bool
        """
        visited = set()
        cur = head
        while cur:
            if cur in visited:
                return True
            visited.add(cur)
            cur = cur.next
        return False