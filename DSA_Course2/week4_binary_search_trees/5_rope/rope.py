import sys


class Rope:
    def __init__(self, s):
        self.s = list(s)  # Convert string to list for mutable operations

    def result(self):
        return ''.join(self.s)  # Convert list back to string

    def process(self, i, j, k):
        # Step 1: Cut the substring from i to j
        substring = self.s[i:j + 1]

        # Step 2: Remove the substring from the original position
        del self.s[i:j + 1]

        # Step 3: Insert the substring at the correct position after k-th symbol
        if k == 0:
            self.s = substring + self.s
        else:
            self.s = self.s[:k] + substring + self.s[k:]


# Read the input
rope = Rope(sys.stdin.readline().strip())  # Read the initial string
q = int(sys.stdin.readline())  # Read the number of queries

# Process each query
for _ in range(q):
    i, j, k = map(int, sys.stdin.readline().strip().split())  # Read query
    rope.process(i, j, k)

# Output the final result
print(rope.result())
