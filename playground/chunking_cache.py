import itertools
import networkx as nx
from collections import Counter
import math

from networkx.classes import DiGraph

def implicant_pattern_to_index(pattern: str) -> int:
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


class Chunk:
    def __init__(self, implicant_pattern: str):
        self.implicant_pattern = implicant_pattern
        self.idx = implicant_pattern_to_index(implicant_pattern)
        self.size = 2**Counter(implicant_pattern)['*']
    def __str__(self):
        return f"Chunk(implicant_pattern={self.implicant_pattern} idx={self.idx} size={self.size})"
    def __repr__(self) -> str:
        return str(self)


def generate_chunks(num_bits: int):
    chunks = [
        Chunk(''.join(chunk))
        for chunk in itertools.product('*-', repeat=num_bits)
    ]
    return {chunk.idx: chunk for chunk in chunks}

def chunk_graph(chunks: list[Chunk]):
    repr_to_index = dict({chunk.implicant_pattern: chunk.idx for chunk in chunks})

    graph = nx.DiGraph()
    for chunk in chunks:
        graph.add_node(chunk.idx)
        for i, c in enumerate(chunk.implicant_pattern):
            if c == '-':
                #continue
                break # we only insert stars to the left of the first dash
            next_chunk = chunk.implicant_pattern[:i] + '-' + chunk.implicant_pattern[i+1:]
            next_chunk_idx = repr_to_index[next_chunk]
            graph.add_edge(chunk.idx, next_chunk_idx)
    return graph

CACHE_SIZE = 12*1024*1024 # 12 MiB, corresponding to i7-1165G7

class Cache:
    def __init__(self):
        self.max_size = CACHE_SIZE
        self.current_size = 0
        self.chunks: list[Chunk] = []
        self.missed_bits = 0

    def access(self, chunk: Chunk):
        #print(chunk, end=' ')
        # hit
        if chunk in self.chunks:
            self.chunks.remove(chunk)
            self.chunks.append(chunk)
            #print("hit")
            return
        # miss
        #print("miss")
        while self.current_size + chunk.size > self.max_size:
            self.current_size -= self.chunks.pop(0).size
        self.current_size += chunk.size
        self.chunks.append(chunk)
        #print(self.chunks)
        self.missed_bits += chunk.size

def evaluate_ordering(graph: DiGraph, chunks: dict[int, Chunk], ordering: list[int]):
    total_bits = sum(chunk.size for chunk in chunks.values())
    cache = Cache()
    for chunk_idx in ordering:
        for node in graph.successors(chunk_idx):
            cache.access(chunks[chunk_idx])
            cache.access(chunks[node])
        cache.access(chunks[chunk_idx])
    return cache.missed_bits - total_bits, total_bits

def get_default_order(graph: DiGraph, chunks: dict[int, Chunk]):
    return sorted(chunks.keys())

def get_dfs_order(graph: DiGraph, chunks: dict[int, Chunk]):
    return list(nx.dfs_preorder_nodes(graph, source=0))

if __name__ == '__main__':
    for i in range(10, 20):
        chunks = generate_chunks(i)
        graph = chunk_graph(list(chunks.values()))

        default_order = get_default_order(graph, chunks)
        capacity_missed_bits, total_bits = evaluate_ordering(graph, chunks, default_order)
        print(f"i={i} default capacity_missed_bits={capacity_missed_bits} total_bits={total_bits} capacity_missed_pct={capacity_missed_bits / total_bits * 100}")

        dfs_order = get_dfs_order(graph, chunks)
        capacity_missed_bits, total_bits = evaluate_ordering(graph, chunks, dfs_order)
        print(f"i={i} dfs_pre capacity_missed_bits={capacity_missed_bits} total_bits={total_bits} capacity_missed_pct={capacity_missed_bits / total_bits * 100}")

