import sys


class Solver:
    def __init__(self, s):
        self.s = s
        self.base1 = 31
        self.mod1 = 10 ** 9 + 7
        self.base2 = 37
        self.mod2 = 10 ** 9 + 9
        self.n = len(s)
        self.prefix_hash1 = [0] * (self.n + 1)
        self.prefix_hash2 = [0] * (self.n + 1)
        self.power1 = [1] * (self.n + 1)
        self.power2 = [1] * (self.n + 1)

        # Precompute the hash values and powers of bases
        for i in range(1, self.n + 1):
            self.prefix_hash1[i] = (self.prefix_hash1[i - 1] * self.base1 + ord(s[i - 1])) % self.mod1
            self.prefix_hash2[i] = (self.prefix_hash2[i - 1] * self.base2 + ord(s[i - 1])) % self.mod2
            self.power1[i] = (self.power1[i - 1] * self.base1) % self.mod1
            self.power2[i] = (self.power2[i - 1] * self.base2) % self.mod2

    def get_hash(self, a, l):
        # Compute the hash for the substring s[a:a+l] using two hash functions
        hash1 = (self.prefix_hash1[a + l] - self.prefix_hash1[a] * self.power1[l]) % self.mod1
        hash2 = (self.prefix_hash2[a + l] - self.prefix_hash2[a] * self.power2[l]) % self.mod2
        if hash1 < 0:
            hash1 += self.mod1
        if hash2 < 0:
            hash2 += self.mod2
        return (hash1, hash2)

    def ask(self, a, b, l):
        return self.get_hash(a, l) == self.get_hash(b, l)


s = sys.stdin.readline().strip()
q = int(sys.stdin.readline())
solver = Solver(s)
for _ in range(q):
    a, b, l = map(int, sys.stdin.readline().split())
    print("Yes" if solver.ask(a, b, l) else "No")
