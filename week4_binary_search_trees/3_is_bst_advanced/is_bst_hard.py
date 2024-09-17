#!/usr/bin/python3

import sys, threading

sys.setrecursionlimit(10 ** 7)  # max depth of recursion
threading.stack_size(2 ** 25)  # new thread will get stack of such size


# Helper function to validate the Binary Search Tree
def is_bst_util(tree, node_index, min_key, max_key):
    if node_index == -1:
        return True  # Empty subtree is valid

    node_key, left_child, right_child = tree[node_index]

    # Check if current node's key is within the valid range
    if not (min_key <= node_key < max_key):
        return False

    # Recursively check the left and right subtrees
    return (is_bst_util(tree, left_child, min_key, node_key) and
            is_bst_util(tree, right_child, node_key, max_key))


def is_binary_search_tree(tree):
    if not tree:
        return True  # Empty tree is valid

    # Start recursive validation from the root (node 0) with the entire range of valid keys
    return is_bst_util(tree, 0, float('-inf'), float('inf'))


def main():
    nodes = int(sys.stdin.readline().strip())
    tree = []

    for i in range(nodes):
        # Reading key, left child, and right child
        tree.append(list(map(int, sys.stdin.readline().strip().split())))

    if is_binary_search_tree(tree):
        print("CORRECT")
    else:
        print("INCORRECT")


threading.Thread(target=main).start()
