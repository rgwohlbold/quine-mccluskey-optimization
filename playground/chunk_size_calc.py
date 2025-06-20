from math import factorial


# # input n
# print("Enter the number of bits n:")
# n = int(input().strip())



# Calculate the chunk size

def choose(n:int, k:int) -> int:
    return factorial(n) // (factorial(k) * factorial(n - k))




for n in range(1, 25):
    print(f"\n=== n = {n} bits ===")
    for i in range(0, n + 1):
        print(f"{i} dashes: {choose(n, i)}")