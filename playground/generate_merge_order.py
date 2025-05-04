from collections import Counter
import itertools
import math

def implicant_pattern_to_index(pattern: str):
    num_bits = len(pattern)
    num_dashes = 0
    section_offset = 0
    for c in pattern:
        if c == '-':
            section_offset += math.comb(num_bits, num_dashes) * (2 ** (num_bits - num_dashes))
            num_dashes += 1
    chunk_offset = 0
    dashes_remaining = num_dashes
    for i in range(num_bits-1, -1, -1):
        if pattern[i] == '-':
            dashes_remaining -= 1
        elif dashes_remaining >= 1:
            chunk_offset += math.comb(i, dashes_remaining-1)
    chunk_offset *= 2 ** (num_bits - num_dashes)

    return section_offset + chunk_offset

def chunk_length(pattern: str):
    return 2 ** Counter(pattern)['*']

def chunk_merge_star_number(input_pattern: str, output_pattern: str):
    if len(input_pattern) != len(output_pattern):
        return False
    difference_index = None
    for i, (ic, oc) in enumerate(zip(input_pattern, output_pattern)):
        if ic == '*' and oc == '-':
            if difference_index is not None:
                return None
            difference_index = i
        elif ic != oc:
            return None
    if difference_index is None:
        return None
    return Counter(input_pattern[difference_index+1:])['*']

def calculate_first_difference(pattern: str):
    num_stars = Counter(pattern)['*']
    num_leading_stars = pattern.find('-')
    if num_leading_stars == -1:
        num_leading_stars = num_stars
    return num_stars - num_leading_stars

def merge_linear(num_bits: int):
    chunks = [
        (''.join(chunk), implicant_pattern_to_index(''.join(chunk)))
        for chunk in itertools.product('*-', repeat=num_bits)
    ]
    chunks.sort(key=lambda x: x[1])
    input_chunk = 0
    output_chunk_left = 1
    output_chunk_right = 1
    difference_indices = []
    while output_chunk_right < len(chunks):
        input_pattern = chunks[input_chunk][0]
        output_pattern = chunks[output_chunk_right][0]
        difference_index = chunk_merge_star_number(input_pattern, output_pattern)
        if difference_index is not None:
            output_chunk_right += 1
            difference_indices.append(difference_index)
        else:
            # we can merge input_chunk into [output_chunk_left, output_chunk_right)
            #if output_chunk_left != output_chunk_right:
            if len(difference_indices) > 0:
                first_difference = min(difference_indices)
            else:
                first_difference = Counter(input_pattern)['*']
            num_stars = Counter(input_pattern)['*']
            print(chunks[input_chunk], chunks[output_chunk_left], chunks[output_chunk_right-1], first_difference)
            assert difference_indices == list(range(first_difference, num_stars))
            assert first_difference == calculate_first_difference(input_pattern)

            output_chunk_left = output_chunk_right
            difference_indices = []
            input_chunk += 1
        if input_chunk >= output_chunk_right:
            raise Exception(f"input_chunk={input_chunk} output_chunk={output_chunk_right}")

if __name__ == '__main__':
    merge_linear(8)
