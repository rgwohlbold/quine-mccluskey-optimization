def quine_mccluskey_step1(bit_width, minterms, dont_cares):
    """
    Perform the first step of the Quine-McCluskey algorithm to find prime implicants.

    Args:
        bit_width (int): Number of bits for the variables.
        minterms (list): List of minterm values.
        dont_cares (list): List of don't care values.

    Returns:
        list: List of prime implicants in binary representation.
    """
    def count_ones(binary):
        return binary.count('1')

    def combine_terms(term1, term2):
        diff_count = 0
        combined = []
        for bit1, bit2 in zip(term1, term2):
            if bit1 != bit2:
                diff_count += 1
                if diff_count > 1:
                    return None
                combined.append('-')
            else:
                combined.append(bit1)
        return ''.join(combined)

    # Combine minterms and don't cares, and convert to binary strings
    all_terms = minterms + dont_cares
    binary_terms = [f"{term:0{bit_width}b}" for term in all_terms]

    # Group terms by the number of 1s
    groups = {}
    for term in binary_terms:
        ones_count = count_ones(term)
        groups.setdefault(ones_count, []).append(term)

    # Combine terms to find prime implicants
    prime_implicants = set()
    while groups:
        new_groups = {}
        used = set()
        for i in sorted(groups.keys()):
            if i + 1 not in groups:
                continue
            for term1 in groups[i]:
                for term2 in groups[i + 1]:
                    combined = combine_terms(term1, term2)
                    if combined:
                        new_groups.setdefault(count_ones(combined), []).append(combined)
                        used.add(term1)
                        used.add(term2)
        # Add unused terms to prime implicants
        for group in groups.values():
            for term in group:
                if term not in used:
                    prime_implicants.add(term)
        groups = new_groups

    sorted_implicants = sorted(prime_implicants)
    
    # Write prime implicants to a file, one per line
    with open('prime_implicants_output.txt', 'w') as f:
        for implicant in sorted_implicants:
            f.write(f"{implicant}\n")
    
    return sorted_implicants

# Example usage
bit_width = 4
minterms = [4, 8, 10, 11, 12, 15]
dont_cares = [9, 14]
prime_implicants = quine_mccluskey_step1(bit_width, minterms, dont_cares)
print("Prime Implicants:", prime_implicants)

