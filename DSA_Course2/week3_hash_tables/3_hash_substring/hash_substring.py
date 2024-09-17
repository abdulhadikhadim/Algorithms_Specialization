def read_input():
    return (input().rstrip(), input().rstrip())


def print_occurrences(output):
    print(' '.join(map(str, output)))


def get_occurrences(pattern, text):
    p = 31  # A prime number used as a base
    m = 10 ** 9 + 7  # A large prime number for modulus
    len_p = len(pattern)
    len_t = len(text)

    # Precompute p^len_p % m
    p_pow = [1] * (len_t)
    for i in range(1, len_t):
        p_pow[i] = (p_pow[i - 1] * p) % m

    # Compute the hash of the pattern
    hash_p = 0
    for i in range(len_p):
        hash_p = (hash_p + (ord(pattern[i]) - ord('a') + 1) * p_pow[i]) % m

    # Compute the hash values of all prefixes of the text
    hash_t = [0] * (len_t + 1)
    for i in range(len_t):
        hash_t[i + 1] = (hash_t[i] + (ord(text[i]) - ord('a') + 1) * p_pow[i]) % m

    # Find all occurrences of the pattern
    occurrences = []
    for i in range(len_t - len_p + 1):
        current_hash = (hash_t[i + len_p] - hash_t[i] + m) % m
        if current_hash == (hash_p * p_pow[i] % m):
            # Double check if the strings are really equal
            if text[i:i + len_p] == pattern:
                occurrences.append(i)

    return occurrences


if __name__ == '__main__':
    print_occurrences(get_occurrences(*read_input()))
