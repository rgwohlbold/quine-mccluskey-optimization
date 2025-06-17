#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Your “measured” times and normalized percentages:
times = [0.001, 0.006, 0.023, 0.057, 0.098, 0.142, 0.193, 0.229, 0.242,
         0.173, 0.118, 0.088, 0.062, 0.046, 0.024, 0.014, 0.006, 0.002,
         0,     0,     0,     0]
sumi = sum(times)
avg_times = [100 * t / sumi for t in times]  # in percent
# ─────────────────────────────────────────────────────────────────────────────

# infer n = len(times)-1
n = len(times) - 1

# compute “theoretical” percentages:
#   f(i) = C(n,i) * 2^(n-i),  total = 3^n
theoretical_pct = []
for i in range(n + 1):
    c = math.comb(n, i)
    p2 = 2 ** (n - i)
    fi = c * p2
    theoretical_pct.append(100.0 * fi / (3 ** n))

# now plot both on the same figure:
xis = list(range(n + 1))
# divide avg times by theoretical
quotient_times = [ math.log(t / theoretical_pct[i]) for i, t in enumerate(avg_times)]
maxi = max(quotient_times)
# normalize quotient_times to [0, 1] range



plt.figure(figsize=(8, 4.5))
plt.plot(xis, avg_times,       marker='o', linestyle='-', label='Observed runtime percentages')
plt.plot(xis, theoretical_pct, marker='s', linestyle='--', label='$\;100\\times \\frac{\\binom{n}{k}2^{\,n-k}}{3^n}$')
plt.plot(xis, quotient_times,  marker='^', linestyle=':', label='Observed / Theoretical')

plt.xlabel('$k$')
plt.ylabel('Percentage [%]')
plt.title(f'Section Percentages for $n={n}$')
plt.xticks(xis)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()

# optionally save to disk:
# plt.savefig("avg_vs_theoretical.png", dpi=300)

plt.show()
