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