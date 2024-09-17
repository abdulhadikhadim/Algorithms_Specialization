def solve(k, text, pattern):
    positions = []
    n, m = len(text), len(pattern)

    mismatches = sum(1 for j in range(m) if text[j] != pattern[j])

    if mismatches <= k:
        positions.append(0)

    for i in range(1, n - m + 1):
        if text[i - 1] != pattern[0]:
            mismatches -= 1
        if text[i + m - 1] != pattern[m - 1]:
            mismatches += 1

        if mismatches <= k:
            positions.append(i)

    return positions

def main():
    while True:
        try:
            line = input().strip()
            if not line:
                break
            k, t, p = line.split()
            ans = solve(int(k), t, p)
            print(len(ans), *ans)
        except EOFError:
            break

if __name__ == "__main__":
    main()
