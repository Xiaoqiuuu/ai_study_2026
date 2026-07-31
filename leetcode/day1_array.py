class Solution_1:
    def twoSum(self, nums: list[int], target:int) -> list[int]:
        dic = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in dic:
                return [dic[complement], i]
            dic[nums[i]] = i

class Solution_26:
    def removeDuplicates(self, nums: list[int]) -> int:
        left, right = 0, 0
        num = nums[0]
        while right < len(nums):
            if nums[left] < nums[right]:
                left = left + 1
                nums[left] = nums[right]
                
            if nums[left] == nums[right]:
                right = right + 1

        return left+1

class Solution_283:
    def moveZeroes(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = 0
        while i < len(nums) :
            if nums[i] == 0:break
            i = i + 1

        if i < len(nums):
            zero_begin, zero_end = i, i
        
            while zero_end < len(nums) - 1 :
                if nums[zero_end + 1] != 0:
                    nums[zero_begin], nums[zero_end + 1] = nums[zero_end + 1], nums[zero_begin]
                    zero_begin += 1
                    zero_end += 1
                else:
                    zero_end = zero_end + 1
                
class Solution_11:
    def maxArea(self, height: list[int]) -> int:
        max_area = 0
        left, right = 0, len(height) - 1
        while left < right:
            area = (right - left) * min(height[left], height[right])
            max_area = max(max_area, area)

            if height[right] > height[left]:
                left = left + 1
            else :
                right = right - 1
        
        return max_area
        



    