#!/usr/bin/env python3
from pathlib import Path
import random


def unwrap(prime: str):
    if prime == "":
        yield ""
    elif prime[0] == "-":
        for v in unwrap(prime[1:]):
            yield "0" + v
            yield "1" + v
    else:
        for v in unwrap(prime[1:]):
            yield prime[0] + v


def parse_file(filename: Path):
    name = None
    num_bits, num_minterms, numprimes = None, None, None
    minterms = []
    primes = []
    lines = filename.read_text().splitlines()
    while lines:
        if lines[0].startswith("#"):
            lines.pop(0)
            continue
        if name is None:
            name = lines.pop(0)
            continue
        if num_bits is None:
            num_bits, num_minterms, numprimes = map(int, (lines.pop(0).split()))
            continue
        if num_minterms > 0:
            minterms.append(int(lines.pop(0)))
            num_minterms -= 1
            continue
        if numprimes > 0:
            primes.append(lines.pop(0))
            numprimes -= 1
            continue
        if num_minterms == 0 and numprimes == 0:
            break
    return (name, num_bits, minterms, primes)


def verify(filename: Path):
    if not filename.exists():
        print(f"File {filename} does not exist")
        return False
    if not filename.is_file():
        print(f"File {filename} is not a file")
        return False
    name, num_bits, minterms, primes = parse_file(filename)

    # print(f"name: {name}", file=sys.stderr)
    # print(f"num_bits: {num_bits}", end=" ", file=sys.stderr)
    # print(f"num_minterms: {len(minterms)}", end=" ", file=sys.stderr)
    # print(f"num_primes: {len(primes)};", file=sys.stderr)
    # Verify test, by unwrapping all the primes and checking if they cover precisely all the minterms
    # Create a list of binary strings

    for p in minterms:
        if not (0 <= p < 2**num_bits):
            print(f"minterm {p} is out of range for {num_bits} bits")
            return False
    for p in primes:
        if len(p) != num_bits:
            print(f"prime {p} is not {num_bits} bits long")
            return False
        for c in p:
            if c not in "01-":
                print(f"prime {p} contains invalid character {c}")
                return False

    implied_minterms = set()
    for p in primes:
        # Unwrap the prime
        implied_minterms.update([int(x, 2) for x in unwrap(p)])
    # print(f"all_primes: {implied_minterms}", file=sys.stderr)
    # print(f"given minterms: {minterms}", file=sys.stderr)
    return implied_minterms == set(minterms) and len(implied_minterms) == len(minterms)


def QMC_naive(minterms: list[int], num_bits: int) -> list[str]:
    level0 = [f"{m:b}".rjust(num_bits, "0") for m in minterms]
    level1 = set()
    merged = set()
    primes = set()
    while True:
        for i in range(len(level0)):
            for j in range(i + 1, len(level0)):
                diffs = []
                for k, (a, b) in enumerate(zip(level0[i], level0[j])):
                    if a != b:
                        diffs.append(k)
                if len(diffs) == 1:
                    # Combine the two minterms
                    merged.add(level0[i])
                    merged.add(level0[j])

                    new_prime = list(level0[i])
                    new_prime[diffs[0]] = "-"
                    new_prime = "".join(new_prime)
                    level1.add(new_prime)
        for l0 in level0:
            if l0 not in merged:
                primes.add(l0)
        # print(f"{level0=}", file=sys.stderr)
        # print(f"  {level1=}", file=sys.stderr)
        # print(f"  {primes=}", file=sys.stderr)
        if len(level1) == 0:
            break
        level0 = list(level1)
        level1 = set()
        merged = set()

    # Convert the primes to the correct format
    primes = sorted(primes)
    for i in range(len(primes)):
        primes[i] = "".join(primes[i])
    return primes


def generate(num_bits: int, density_percent: float, seed=None):
    if seed is not None:
        random.seed(seed)
    minterm_set = set()
    num_true = int((2**num_bits) * (density_percent / 100))
    while len(minterm_set) < num_true:
        minterm_set.add(random.randint(0, 2**num_bits - 1))
    minterms = sorted(minterm_set)
    primes = QMC_naive(minterms, num_bits)
    print(
        f"rnd-{num_bits}-{density_percent:.0f}pct-s{seed if seed is not None else 'rnd'}.qmc"
    )
    print(f"# {num_bits} bits")
    print(f"{num_bits} {len(minterms)} {len(primes)}")
    print("# Minterms")
    for m in minterms:
        print(m)
    print("# Primes")
    for p in primes:
        print(p)
    print(f"# minterms: {len(minterms)}/{int(2**num_bits)}, primes: {len(primes)}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    cmdp = parser.add_subparsers(title="commands", dest="command")

    verifyp = cmdp.add_parser("verify")
    verifyp.add_argument("file", type=Path, help="File to validate over")
    generatep = cmdp.add_parser("generate")
    generatep.add_argument("num_bits", type=int, help="Number of bits function takes")
    generatep.add_argument(
        "pct_density", type=float, help="Percent of minterms (out of 2^n)"
    )
    generatep.add_argument(
        "-s", "--seed", type=int, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    if args.command == "verify":
        f = args.file
        if not verify(Path(f)):
            print(f"{f}\tINVALID")
            sys.exit(1)
        print(f"{f}\tVALID")
    elif args.command == "generate":
        if args.num_bits < 1 or args.num_bits > 30:
            print("Number of bits must be between 1 and 30", file=sys.stderr)
            sys.exit(1)
        if args.pct_density < 0 or args.pct_density > 100:
            print("Density percent must be between 0 and 100", file=sys.stderr)
            sys.exit(1)
        generate(args.num_bits, args.pct_density, seed=args.seed)
        print(
            f"Generating {args.pct_density}% density for {args.num_bits} bits with seed {args.seed}",
            file=sys.stderr,
        )
    else:
        print("No command given")
        parser.print_help()
        sys.exit(1)
    sys.exit(0)
