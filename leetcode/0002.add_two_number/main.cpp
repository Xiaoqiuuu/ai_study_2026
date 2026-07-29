#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* cur = &dummy;
        int carry = 0;

        while (l1 || l2 || carry) {
            int sum = carry;
            if (l1) {
                sum += l1->val;
                l1 = l1->next;
            }
            if (l2) {
                sum += l2->val;
                l2 = l2->next;
            }

            carry = sum / 10;
            cur->next = new ListNode(sum % 10);
            cur = cur->next;
        }

        return dummy.next;
    }
};

void printList(ListNode* head) {
    while (head) {
        std::cout << head->val;
        if (head->next) {
            std::cout << " -> ";
        }
        head = head->next;
    }
    std::cout << std::endl;
}

int main() {
    Solution solution;
    ListNode* l1 = new ListNode(2, new ListNode(4, new ListNode(3)));
    ListNode* l2 = new ListNode(5, new ListNode(6, new ListNode(4)));
    ListNode* result = solution.addTwoNumbers(l1, l2);

    std::cout << "2 -> 4 -> 3 + 5 -> 6 -> 4 = ";
    printList(result);

    ListNode* l3 = new ListNode(9, new ListNode(9, new ListNode(9)));
    ListNode* l4 = new ListNode(1);
    ListNode* result2 = solution.addTwoNumbers(l3, l4);

    std::cout << "9 -> 9 -> 9 + 1 = ";
    printList(result2);

    return 0;
}