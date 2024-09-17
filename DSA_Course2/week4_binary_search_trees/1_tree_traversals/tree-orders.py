import sys
import threading

sys.setrecursionlimit(10**6)
threading.stack_size(2**27)

class TreeOrders:
    def read(self):
        self.n = int(input())  # Use input() instead of sys.stdin.readline() for better compatibility
        self.key = [0] * self.n
        self.left = [0] * self.n
        self.right = [0] * self.n
        for i in range(self.n):
            a, b, c = map(int, input().split())
            self.key[i] = a
            self.left[i] = b
            self.right[i] = c

    def inOrderTraversal(self, index):
        if index == -1:
            return
        self.inOrderTraversal(self.left[index])
        self.result.append(self.key[index])
        self.inOrderTraversal(self.right[index])

    def preOrderTraversal(self, index):
        if index == -1:
            return
        self.result.append(self.key[index])
        self.preOrderTraversal(self.left[index])
        self.preOrderTraversal(self.right[index])

    def postOrderTraversal(self, index):
        if index == -1:
            return
        self.postOrderTraversal(self.left[index])
        self.postOrderTraversal(self.right[index])
        self.result.append(self.key[index])

    def inOrder(self):
        self.result = []
        self.inOrderTraversal(0)
        return self.result

    def preOrder(self):
        self.result = []
        self.preOrderTraversal(0)
        return self.result

    def postOrder(self):
        self.result = []
        self.postOrderTraversal(0)
        return self.result

def main():
    tree = TreeOrders()
    tree.read()
    print(" ".join(map(str, tree.inOrder())))
    print(" ".join(map(str, tree.preOrder())))
    print(" ".join(map(str, tree.postOrder())))

threading.Thread(target=main).start()
