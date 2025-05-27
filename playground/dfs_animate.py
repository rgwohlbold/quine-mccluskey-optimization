import re, networkx as nx, matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation

"""
How to use:
- Compile binary with -DLOG_LEVEL=3 and pipe output to log file.
- Update N to1

"""


def prod(i: iter):
    """Returns the product of all elements in the iterable."""
    p = 1
    for x in i:
        p *= x
    return p


def comb(n: int, k: int):
    """Returns the number of combinations of n items taken k at a time."""
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return prod(range(n - k + 1, n + 1)) // prod(range(1, k + 1))


def tree_level_width(n: int, depth: int):
    return comb(n, depth)


def parse_line(lin: str):
    m = re.search(r"Section OUT\s+(\d+); \[inp=(\d+)/out=(\d+)/all=\d+\]", lin)
    if not m:
        return None

    return int(m.group(1)), int(m.group(2)), int(m.group(3))


with open(sys.argv[1]) as f:
    LOGLINES = f.readlines()

# Find the line with num_bits=N
num_b_re = re.compile(r"num_bits=(\d+)\s+num_trues=")
for l in LOGLINES:
    if (m := num_b_re.search(l)) is not None:
        N = int(m.group(1))
        break
else:
    print("No num_bits found in log file.")
    sys.exit(1)
print("N: ", N)
max_n = 11
assert N <= max_n, f"N must be less than or equal to {max_n} for visualization purposes."

layer_nodes = {k: [] for k in range(N + 1)}
G = nx.DiGraph()
for layer in range(N + 1):
    for i in range(tree_level_width(N, layer)):
        G.add_node((layer, i))
        G.nodes[(layer, i)]["color"] = "red"
        layer_nodes[layer].append((layer, i))

pos = nx.spring_layout(G)
events = [e for line in LOGLINES if ((e := parse_line(line)) is not None)]
print("Num of events: ", len(events))
fig, ax = plt.subplots()

def pos_func(node):
    layer, index = node
    return (index * 2, -layer)  # Offset by index and layer depth


pos = {node: pos_func(node) for node in G.nodes()}


def update(i):
    ax.clear()
    evt = events[i]
    layer, green_to, red_to = evt
    # print(f"Processing event {i}: Layer {layer}, Green to {green_to}, Red to {red_to}")
    for j in range(len(layer_nodes[layer])):
        node = layer_nodes[layer][j]
        if j < green_to:
            color = "green"
        elif j < red_to:
            color = "yellow"
        else:
            color = "red"
        G.nodes[node]["color"] = color
    nx.draw(
        G,
        pos,
        node_color=[G.nodes[node]["color"] for node in G.nodes()],
    )

anim = FuncAnimation(fig, update, frames=len(events), interval=50)
plt.show()
