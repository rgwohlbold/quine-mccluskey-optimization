import pandas as pd

# determined by setting COUNT_OPS=1
num_ops={1:5,2:29,3:127,4:495,5:1811,6:6375,7:21883,8:73823,9:245923,10:811383,11:2656523,12:8642319,13:27962611,14:90043943,15:288731227,16:922327743,17:2936220995,18:9318407895,19:29489308075}

df = pd.read_csv("measurements.csv")
df = df[df['implementation'] == 'prime_implicants_dense']
df['ops'] = df['bits'].map(num_ops)
df['performance'] = df['ops'] / df['cycles']
print(df)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 5))
sns.lineplot(data=df, x='bits', y='performance', marker='o')

plt.xlabel('n')
plt.ylabel('Performance [ops/cycle]')
plt.title('Performance of prime_implicants_dense for different n')
plt.grid(True)
plt.autoscale(enable=True, axis='y', tight=False)
plt.ylim(bottom=0)
plt.savefig("performance.png", dpi=300, bbox_inches="tight")

plt.show()