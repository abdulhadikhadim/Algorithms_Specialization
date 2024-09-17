#!/usr/bin/python3

import sys, threading

sys.setrecursionlimit(10 ** 7)  # Max depth of recursion
threading.stack_size(2 ** 25)  # New thread will get a stack of such size


# A helper function to determine if the tree is a valid binary search tree
def IsBinarySearchTree(tree):
    # If there's no tree, it's considered valid
    if len(tree) == 0:
        return True

    # Recursive function to check the subtree rooted at 'node_index'
    def is_bst(node_index, min_key, max_key):
        if node_index == -1:  # Base case: empty subtree
            return True

        key, left, right = tree[node_index]

        # Check if the current node's key violates the min/max constraints
        if key <= min_key or key >= max_key:
            return False

        # Recursively check the left and right subtrees with updated constraints
        return is_bst(left, min_key, key) and is_bst(right, key, max_key)

    # Start checking from the root node (0), with an infinite range for valid keys
    return is_bst(0, float('-inf'), float('inf'))


def main():
    nodes = int(sys.stdin.readline().strip())
    tree = []

    for i in range(nodes):
        tree.append(list(map(int, sys.stdin.readline().strip().split())))

    if IsBinarySearchTree(tree):
        print("CORRECT")
    else:
        print("INCORRECT")


# Using threading to increase the stack size and prevent recursion limit issues
threading.Thread(target=main).start()
