import sys
if __name__ == '__main__':
    num_bits = int(sys.argv[1])
    num_minterms = 2**num_bits
    filename = f"tests/all_minterms_{num_bits}.txt"
    with open(filename, "w") as f:
        f.write(f"all minterms {num_bits}\n")
        f.write(f"{num_bits} {num_minterms} 1\n")
        for i in range(num_minterms):
            f.write(f"{i}\n")
        f.write("-"*num_bits + "\n")
