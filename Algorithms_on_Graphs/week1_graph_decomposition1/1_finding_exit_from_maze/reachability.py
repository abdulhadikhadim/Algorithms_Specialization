def dfs(adj, visited, node):
    visited[node] = True
    for neighbor in adj[node]:
        if not visited[neighbor]:
            dfs(adj, visited, neighbor)


def reach(adj, x, y):
    visited = [False] * len(adj)
    dfs(adj, visited, x)
    return 1 if visited[y] else 0


if __name__ == '__main__':
    # Reading inputs
    n, m = map(int, input().split())  # number of vertices and edges
    adj = [[] for _ in range(n)]

    # Reading all edges
    for _ in range(m):
        u, v = map(int, input().split())
        u, v = u - 1, v - 1  # zero-indexing
        adj[u].append(v)
        adj[v].append(u)

    # Reading the vertices x and y
    x, y = map(int, input().split())
    x, y = x - 1, y - 1  # zero-indexing

    # Output the result of reachability check
    print(reach(adj, x, y))
