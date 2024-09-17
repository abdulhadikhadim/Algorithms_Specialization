def rabin_karp(s, t, length):
    # Prime number for modulus
    p = 10 ** 9 + 7
    # Base value for the hash function
    x = 263

    s_len = len(s)
    t_len = len(t)

    # Compute the hash of the first substring of given length
    s_hashes = {}
    t_hashes = {}

    # Compute the hash of the first substring of length `length`
    s_hash = 0
    t_hash = 0
    x_l = pow(x, length, p)  # x^length % p

    for i in range(length):
        s_hash = (s_hash * x + ord(s[i])) % p
        t_hash = (t_hash * x + ord(t[i])) % p

    s_hashes[s_hash] = [0]
    t_hashes[t_hash] = [0]

    # Compute the hashes of the subsequent substrings
    for i in range(1, s_len - length + 1):
        s_hash = (s_hash * x + ord(s[i + length - 1]) - ord(s[i - 1]) * x_l) % p
        if s_hash not in s_hashes:
            s_hashes[s_hash] = []
        s_hashes[s_hash].append(i)

    for i in range(1, t_len - length + 1):
        t_hash = (t_hash * x + ord(t[i + length - 1]) - ord(t[i - 1]) * x_l) % p
        if t_hash not in t_hashes:
            t_hashes[t_hash] = []
        t_hashes[t_hash].append(i)

    # Find the common hash value between the two strings
    for h in s_hashes:
        if h in t_hashes:
            for i in s_hashes[h]:
                for j in t_hashes[h]:
                    if s[i:i + length] == t[j:j + length]:
                        return i, j, length
    return None


def solve(s, t):
    left, right = 0, min(len(s), len(t))
    result = (0, 0, 0)

    while left <= right:
        mid = (left + right) // 2
        common_substring = rabin_karp(s, t, mid)
        if common_substring:
            result = common_substring
            left = mid + 1
        else:
            right = mid - 1

    return result


# Read multiple lines of input from the user
input_lines = []
while True:
    try:
        line = input().strip()
        if line:
            input_lines.append(line)
        else:
            break
    except EOFError:
        break

# Process each line of input
for line in input_lines:
    s, t = line.split()
    ans = solve(s, t)
    # Output the result
    print(ans[0], ans[1], ans[2])
