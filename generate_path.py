#!/usr/bin/env python3
import math
import argparse
import os
import sys
import struct

def binomial_coefficient(n, k):
    return math.comb(n, k)

def leading_stars(num_bits, num_dashes, chunk_index):
    dashes_passed = 0
    for i in range(num_bits):
        dashes_remaining = num_dashes - dashes_passed
        if dashes_remaining == 0:
            return num_bits - i
        count_if_dash = binomial_coefficient(num_bits - i - 1, dashes_remaining - 1)
        if chunk_index < count_if_dash:
            dashes_passed += 1
        else:
            chunk_index -= count_if_dash
    return 0

def generate_flat_schedule(n, out_filename):
    with open(out_filename, 'wb') as f:
        input_index = 0
        for num_dashes in range(0, n + 1):
            rem_bits = n - num_dashes
            iterations = binomial_coefficient(n, num_dashes)
            input_elems = 1 << rem_bits
            output_elems = (1 << (rem_bits - 1)) if rem_bits > 0 else 0
            output_index = input_index + iterations * input_elems

            for chunk_idx in range(iterations):
                ls = leading_stars(n, num_dashes, chunk_idx)
                first_diff = rem_bits - ls
                f.write(struct.pack("<BBQQ", rem_bits, first_diff, input_index, output_index))
                output_index += (rem_bits - first_diff) * output_elems
                input_index += input_elems

    print(f"[flat] wrote merge schedule for n={n} → {out_filename}", file=sys.stderr)

def generate_dfs_schedule(n, out_filename):
    print(n)
    total_chunks     = [binomial_coefficient(n, d) for d in range(n + 1)]
    base_section_off = [0] * (n + 1)
    for d in range(1, n + 1):
        prev = base_section_off[d - 1]
        rem   = n - d
        base_section_off[d] = prev + total_chunks[d - 1] * (1 << rem + 1)

    in_stack  = [0] * (n + 1)
    out_stack = [0] * (n + 1)
    out_stack[0] = 1

    section = 0
    print(base_section_off)
    with open(out_filename, 'wb') as f:
        while section >= 0:
            in_chunk  = in_stack[section]
            out_chunk = out_stack[section]
            total     = total_chunks[section]
            if section == n or in_chunk >= total or out_chunk <= in_chunk:
                section -= 1
                continue

            rem_bits   = n - section
            ls         = leading_stars(n, section, in_chunk)
            first_diff = rem_bits - ls
            inp_idx    = base_section_off[section]   + in_chunk  * (1 << rem_bits)
            out_idx    = base_section_off[section+1] + out_stack[section+1] * (1 << (rem_bits - 1))

            f.write(struct.pack("<BBQQ", rem_bits, first_diff, inp_idx, out_idx))

            in_stack[section]      += 1
            out_stack[section + 1] += ls

            if section == n:
                section -= 1
            else:
                section += 1

    print(f"[dfs] wrote merge schedule for n={n} → {out_filename}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Emit prime-implicant merge schedules up to a given bit-width"
    )
    parser.add_argument(
        "max_bits", type=int,
        help="Maximum bit-width (will write schedules for n=1..max_bits)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["flat","dfs"], default="flat",
        help="Which traversal to emit (flat nested‐loops, or dfs)"
    )
    parser.add_argument(
        "--out-dir", "-o", default="traversals",
        help="Directory in which to write the schedule files"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for n in range(1, args.max_bits + 1):
        fname = f"merge_schedule_n{n}_{args.mode}.txt"
        path  = os.path.join(args.out_dir, fname)
        print(f"Generating {args.mode} schedule for n={n} → {path}", file=sys.stderr)
        if args.mode == "flat":
            generate_flat_schedule(n, path)
        else:
            generate_dfs_schedule(n, path)

if __name__ == "__main__":
    main()
