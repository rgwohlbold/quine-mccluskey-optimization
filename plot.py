import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("measurements.csv")
df = df.groupby(['compiler_version', 'compiler_flags', 'cpu_model', 'implementation', 'bits']).median().reset_index()
df['performance'] = df['ops'] / df['cycles']
df['function'] = df['implementation'] + ', ' + df['compiler_version'] + ', ' + df['compiler_flags']
print(df)

for i, df_grouped in df.groupby(['cpu_model']):
    df_reset = df_grouped.reset_index()
    cpu_model = df_reset['cpu_model'][0]
    compiler_version = df_reset['compiler_version'][0]
    compiler_flags = df_reset['compiler_flags'][0]

    plt.figure(figsize=(9, 6))
    sns.lineplot(data=df_grouped, x='bits', y='performance', hue='function', marker='o')

    plt.xlabel('n')
    plt.ylabel('Performance [ops/cycle]')
    plt.title(f"Performance of all implementations for different number of bits n\nCPU: {cpu_model}", loc='left')
    plt.grid(True)
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.ylim(bottom=0)
    plt.xticks(df_grouped['bits'].unique())
    plt.savefig("performance.png", dpi=300, bbox_inches="tight")

    plt.show()
