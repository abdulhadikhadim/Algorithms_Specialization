def number_of_components(adj):
    def dfs(v):
        visited[v] = True
        for neighbor in adj[v]:
            if not visited[neighbor]:
                dfs(neighbor)

    n = len(adj)
    visited = [False] * n
    result = 0

    for i in range(n):
        if not visited[i]:
            dfs(i)
            result += 1

    return result


if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(m)]
    adj = [[] for _ in range(n)]

    for (a, b) in edges:
        adj[a - 1].append(b - 1)
        adj[b - 1].append(a - 1)

    print(number_of_components(adj))
