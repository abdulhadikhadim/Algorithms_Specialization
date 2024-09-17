import sys
import threading


def compute_height(n, parents):
    # Array to store the height of each node
    heights = [0] * n

    def node_height(vertex):
        # If height is already computed, return it
        if heights[vertex] != 0:
            return heights[vertex]

        # If the vertex is the root
        if parents[vertex] == -1:
            heights[vertex] = 1
            return heights[vertex]

        # Recursively compute the height of the parent and then add 1
        parent_height = node_height(parents[vertex])
        heights[vertex] = parent_height + 1
        return heights[vertex]

    # Compute the height of each node
    for vertex in range(n):
        node_height(vertex)

    # The height of the tree is the maximum height among all nodes
    return max(heights)


def main():
    n = int(input())
    parents = list(map(int, input().split()))
    print(compute_height(n, parents))


sys.setrecursionlimit(10 ** 7)
threading.stack_size(2 ** 27)
threading.Thread(target=main).start()
