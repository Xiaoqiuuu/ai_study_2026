# 200, 207, 695
class Solution_200:
    def numIslands(self, grid: list[list[str]]) -> int:
        def pad_with_zeros(matrix):
            if not matrix or not matrix[0]:
                return matrix
            rows = len(matrix)
            cols = len(matrix[0])
            padded = [['0'] * (cols + 2)]                     
            padded += [['0'] + row + ['0'] for row in matrix]   
            padded += [['0'] * (cols + 2)]                   
            return padded
        
        grid_padded = pad_with_zeros(grid)
        m = len(grid_padded)
        n = len(grid_padded[0])
        
        vis = [[0] * n for _ in range(m)]
        num = 0

        def search(i: int, j:int):
            if vis[i][j] == 1 or grid_padded[i][j] == '0':return 
            vis[i][j] = 1
            search(i + 1, j)
            search(i, j + 1)
            search(i - 1, j)
            search(i, j - 1)
            

        i, j = 1, 1
        while i < m-1:
            while j < n-1:
                if vis[i][j] == 0 and grid_padded[i][j] == '1':
                    num += 1
                    search(i, j) 
                j += 1
            j = 1
            i += 1
        return num   

class solution_207:
    def canFinish(self, numCourses, prerequisites):
        in_degrees = [0 for _ in range(numCourses)]  

        #邻接表，存放的是该课程指向的课程集合
        nextCourse = [set() for _ in range(numCourses)]  

        for second, first in prerequisites:
            in_degrees[second] += 1   
            nextCourse[first].add(second)   

        zeroQ = []  
        for i in range(numCourses):
            if in_degrees[i] == 0:
                zeroQ.append(i)

        count = 0  
        while zeroQ:
            n = zeroQ.pop()
            count += 1
            for c in nextCourse[n]:
                in_degrees[c] -= 1
                if in_degrees[c] == 0:  
                    zeroQ.append(c)

        return count == numCourses


class Solution_695:
    def maxAreaOfIsland(self, grid: list[list[int]]) -> int:
        def pad_with_zeros(matrix):
                if not matrix or not matrix[0]:
                    return matrix
                rows = len(matrix)
                cols = len(matrix[0])
                padded = [['0'] * (cols + 2)]                     
                padded += [['0'] + row + ['0'] for row in matrix]   
                padded += [['0'] * (cols + 2)]                   
                return padded
            
        grid_padded = pad_with_zeros(grid)
        m = len(grid_padded)
        n = len(grid_padded[0])
        
        vis = [[0] * n for _ in range(m)]
        maxarea = 0

        def search(i: int, j:int):
            nonlocal area
            if vis[i][j] == 1 or grid_padded[i][j] == '0':return 
            vis[i][j] = 1
            area += 1
            search(i + 1, j)
            search(i, j + 1)
            search(i - 1, j)
            search(i, j - 1)
            

        i, j = 1, 1
        while i < m-1:
            while j < n-1:
                if vis[i][j] == 0 and grid_padded[i][j] == '1':
                    area = 0
                    search(i, j) 
                    maxarea = max(area, maxarea)
                j += 1
            j = 1
            i += 1
        return maxarea