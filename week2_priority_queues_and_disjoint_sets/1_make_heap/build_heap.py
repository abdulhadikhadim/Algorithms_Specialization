# python3

def build_heap(data):
    """Build a heap from ``data`` inplace using sift down.

    Returns a sequence of swaps performed by the algorithm.
    """
    swaps = []
    n = len(data)

    # Start from the last non-leaf node and sift down each node
    for i in range(n // 2 - 1, -1, -1):
        sift_down(data, i, swaps)

    return swaps


def sift_down(data, i, swaps):
    """Sifts down the element at index i in the heap."""
    min_index = i
    left_child_index = 2 * i + 1
    right_child_index = 2 * i + 2

    # Check if the left child exists and is smaller than the current element
    if left_child_index < len(data) and data[left_child_index] < data[min_index]:
        min_index = left_child_index

    # Check if the right child exists and is smaller than the current minimum
    if right_child_index < len(data) and data[right_child_index] < data[min_index]:
        min_index = right_child_index

    # If the smallest element is not the current element, swap and continue sifting down
    if i != min_index:
        swaps.append((i, min_index))
        data[i], data[min_index] = data[min_index], data[i]
        sift_down(data, min_index, swaps)


def main():
    n = int(input())
    data = list(map(int, input().split()))
    assert len(data) == n

    swaps = build_heap(data)

    print(len(swaps))
    for i, j in swaps:
        print(i, j)


if __name__ == "__main__":
    main()
