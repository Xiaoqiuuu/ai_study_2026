#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        vector<int> merged;
        merged.reserve(m + n);
        int i = 0, j = 0;
        while (i < m && j < n) {
            if (nums1[i] < nums2[j]) merged.push_back(nums1[i++]);
            else merged.push_back(nums2[j++]);
        }
        while (i < m) merged.push_back(nums1[i++]);
        while (j < n) merged.push_back(nums2[j++]);

        int sz = m + n;
        if (sz == 0) return 0.0;
        if (sz % 2 == 1) return merged[sz/2];
        return (merged[sz/2 - 1] + merged[sz/2]) / 2.0;
    }
};

int main(){
    vector<int> num1 = {1,3};
    vector<int> num2 = {2};
    Solution sol;
    double ans = sol.findMedianSortedArrays(num1,num2);
    cout << ans << endl;
    return 0;
}