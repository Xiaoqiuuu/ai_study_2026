#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string convert(string s, int numRows) {
        int n = s.length();
        int numCols = n / numRows + 1;
        int num[n];
        for (int i = 0; i < n; i++){
            int cycleLen = 2 * numRows - 2;
            int posInCycle = i % cycleLen;
            num[i] = posInCycle;
        }
        char result[numRows][numCols];
        for (int i = 0; i < numRows; i++){
            for (int j = 0; j < numCols; j++){
                result[i][j] = ' ';
            }
        }
        int row = 0, col = 0;
        for (int i = 0; i < n; i++){
            result[row][col] = s[i];
            if (num[i] < numRows){
                row++;
            } else {
                row--;
                col++;
            }
        }
        string converted = "";
        for (int i = 0; i < numRows; i++){
            for (int j = 0; j < numCols; j++){
                if (result[i][j] != ' '){
                    converted += result[i][j];
                }
            }
        }
        return converted;
    }
};