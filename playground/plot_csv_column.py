#!/usr/bin/env python3
import pandas as pd
import sys

# Usage: python plot_csv.py <csv_file> <column>
if len(sys.argv) != 3:
    print("Usage: python plot_csv.py <csv_file> <column>")
    sys.exit(1)
df = pd.read_csv(sys.argv[1], header=0)
print(df)

col_name = sys.argv[2]
# Group by "cmd", take its median, show 1std for all the runs

if col_name not in df.columns:
    print(f"Column '{col_name}' not found in the CSV file.")
    sys.exit(1)
grouped = (
    df.groupby(["cmd", "nbits"])[col_name].agg(["mean", "std", "count"]).reset_index()
)

# Plot line plot with error bars, for each cmd it's own line
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for cmd, group in grouped.groupby("cmd"):
    plt.errorbar(
        group["nbits"],
        group["mean"],
        yerr=group["std"],
        label=cmd,
        marker="o",
        capsize=5,
    )

plt.title(f"{col_name} by cmd and nbits")
plt.xlabel("Number of bits (nbits)")
plt.ylabel(col_name)
plt.xticks(grouped["nbits"].unique())
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
