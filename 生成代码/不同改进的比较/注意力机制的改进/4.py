def max_moves(matrix, n, m):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[False] * m for _ in range(n)]

    def is_valid(x, y):
        return 0 <= x < n and 0 <= y < m

    def dfs(x, y, target):
        if matrix[x][y] != target:
            return 0
        if visited[x][y]:
            return -1

        visited[x][y] = True
        max_moves_count = 0

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if is_valid(new_x, new_y):
                next_moves_count = dfs(new_x, new_y, chr(ord(target) + 1))
                if next_moves_count == -1:
                    return -1
                max_moves_count = max(max_moves_count, 1 + next_moves_count)

        visited[x][y] = False
        return max_moves_count

    max_moves_count = 0

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == "A":
                moves_count = dfs(i, j, "A")
                if moves_count == -1:
                    return -1
                max_moves_count = max(max_moves_count, moves_count)

    return max_moves_count

n, m = map(int, input().split())
matrix = [list(input()) for _ in range(n)]

result = max_moves(matrix, n, m)
if result == 0:
    print(-1)
else:
    print(result)
