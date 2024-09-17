from sys import stdin

# Splay tree implementation

class Vertex:
    def __init__(self, key, sum_val, left=None, right=None, parent=None):
        self.key = key
        self.sum_val = sum_val
        self.left = left
        self.right = right
        self.parent = parent

def update(v):
    if v is None:
        return
    v.sum_val = v.key + (v.left.sum_val if v.left else 0) + (v.right.sum_val if v.right else 0)
    if v.left:
        v.left.parent = v
    if v.right:
        v.right.parent = v

def smallRotation(v):
    parent = v.parent
    if parent is None:
        return
    grandparent = parent.parent
    if parent.left == v:
        m = v.right
        v.right = parent
        parent.left = m
    else:
        m = v.left
        v.left = parent
        parent.right = m
    update(parent)
    update(v)
    v.parent = grandparent
    if grandparent:
        if grandparent.left == parent:
            grandparent.left = v
        else:
            grandparent.right = v

def bigRotation(v):
    if v.parent.left == v and v.parent.parent.left == v.parent:
        smallRotation(v.parent)
        smallRotation(v)
    elif v.parent.right == v and v.parent.parent.right == v.parent:
        smallRotation(v.parent)
        smallRotation(v)
    else:
        smallRotation(v)
        smallRotation(v)

def splay(v):
    if v is None:
        return None
    while v.parent:
        if v.parent.parent is None:
            smallRotation(v)
        else:
            bigRotation(v)
    return v

def find(root, key):
    v = root
    last = root
    next_node = None
    while v:
        if v.key >= key and (next_node is None or v.key < next_node.key):
            next_node = v
        last = v
        v = v.right if v.key < key else v.left
    root = splay(last)
    return next_node, root

def split(root, key):
    result, root = find(root, key)
    if result is None:
        return root, None
    right = splay(result)
    left = right.left
    right.left = None
    if left:
        left.parent = None
    update(left)
    update(right)
    return left, right

def merge(left, right):
    if left is None:
        return right
    if right is None:
        return left
    while right.left:
        right = right.left
    right = splay(right)
    right.left = left
    update(right)
    return right

def insert(x):
    global root
    left, right = split(root, x)
    if right is None or right.key != x:
        new_vertex = Vertex(x, x)
        root = merge(merge(left, new_vertex), right)
    else:
        root = merge(left, right)

def erase(x):
    global root
    result, root = find(root, x)
    if result and result.key == x:
        root = splay(result)
        root = merge(root.left, root.right)
        if root:
            root.parent = None

def search(x):
    global root
    result, root = find(root, x)
    return result and result.key == x

def sum(fr, to):
    global root
    left, middle = split(root, fr)
    middle, right = split(middle, to + 1)
    ans = 0 if middle is None else middle.sum_val
    root = merge(merge(left, middle), right)
    return ans

# Main logic
MODULO = 1000000001
n = int(stdin.readline())
last_sum_result = 0
root = None

for _ in range(n):
    line = stdin.readline().split()
    if line[0] == '+':
        x = int(line[1])
        insert((x + last_sum_result) % MODULO)
    elif line[0] == '-':
        x = int(line[1])
        erase((x + last_sum_result) % MODULO)
    elif line[0] == '?':
        x = int(line[1])
        print('Found' if search((x + last_sum_result) % MODULO) else 'Not found')
    elif line[0] == 's':
        l = int(line[1])
        r = int(line[2])
        res = sum((l + last_sum_result) % MODULO, (r + last_sum_result) % MODULO)
        print(res)
        last_sum_result = res % MODULO
