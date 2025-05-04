import pandas as pd

df = pd.read_csv("measurements.csv")
df = df[df['implementation'] == 'baseline']
df = df.groupby(['implementation', 'bits']).median().reset_index()
df['performance'] = df['ops'] / df['cycles']
print(df)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))
sns.lineplot(data=df, x='bits', y='performance', marker='o')

plt.xlabel('n')
plt.ylabel('Performance [ops/cycle]')
plt.title('Performance of baseline for different n')
plt.grid(True)
plt.autoscale(enable=True, axis='y', tight=False)
plt.ylim(bottom=0)
plt.xticks(df['bits'].unique())
plt.savefig("performance.png", dpi=300, bbox_inches="tight")

plt.show()