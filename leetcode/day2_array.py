class Solution_42:
    def trap(self, height: list[int]) -> int:
        if not height:
            return 0
        
        left = 0
        right = []
        area = 0
        i = 0
        
        while i < len(height):
            if not right or height[i] <= height[right[-1]]:
                right.append(i)
                i += 1
            else:
                bottom = right.pop()
                if right:
                    left = right[-1]
                    h = min(height[left], height[i]) - height[bottom]
                    w = i - left - 1
                    area += h * w
        
        return area

class Solution_3:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_set = set()
        left = 0
        max_length = 0

        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)
        return max_length

class Solution_704:
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1

class Solution_75:
    def sortColors(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        low, mid, high = 0, 0, len(nums) - 1

        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:  # nums[mid] == 2
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
class Solution_560:
    def subarraySum(self, nums: list[int], k: int) -> int:
        #前缀和 不会
        