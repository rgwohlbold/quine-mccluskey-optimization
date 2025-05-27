import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def performance_plot(df):
    cpu_model = df['cpu_model'][0]
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=df, x='bits', y='performance', hue='function', marker='o')

    plt.xlabel('n')
    plt.ylabel('Performance [ops/cycle]')
    plt.title(f"Performance of all merge step implementations for different number of bits n\nCPU: {cpu_model}", loc='left')
    plt.grid(True)
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.ylim(bottom=0)
    plt.xticks(df['bits'].unique())
    plt.savefig("merge_performance.png", dpi=300, bbox_inches="tight")

    plt.show()

def calculate_bytes(n):
    BYTES_PER_BOOL = 0.125 # we use bitmasks, so 8 bools per byte
    input_bools = 2**n
    merge_bools = 2**n
    output_bools = n*2**(n-1)
    return (input_bools+merge_bools+output_bools)*BYTES_PER_BOOL

# Source: Gemini 2.5 Pro
def roofline_plot(df):
    # TODO: use hw values here
    PI_PEAK = 3 * 256.0 # Skylake: throughput of 3 for _mm256_and_si256 -> 3*356 logic operations per second
    BETA_PEAK = 30.0

    # Plot the measured performance points
    sns.lineplot(data=df, x='operational_intensity', y='performance', hue='function', marker='o')
    # Annotate points with 'n' values
    for i in range(df.shape[0]):
        plt.text(df['operational_intensity'].iloc[i],
                df['performance'].iloc[i],
                f' n={df["bits"].iloc[i]}',
                fontsize=9, verticalalignment='bottom', horizontalalignment='left')
    min_intensity = df['operational_intensity'].min()
    max_intensity = df['operational_intensity'].max()
    max_perf = df['performance'].max()

    # Extend plot range slightly for visibility
    plot_min_intensity = min_intensity / 2
    plot_max_intensity = max_intensity * 2
    plot_max_perf = max(max_perf, PI_PEAK) * 1.5

    # Calculate roofline points
    # Create a range of operational intensity values for plotting the roof
    intensity_range = np.logspace(np.log10(plot_min_intensity), np.log10(plot_max_intensity), 100)

    # Memory-bound roof (Performance = Beta * Intensity)
    memory_roof = BETA_PEAK * intensity_range

    # Compute-bound roof (Performance = Pi_peak)
    compute_roof = np.full_like(intensity_range, PI_PEAK)

    # Actual roofline is the minimum of the two
    roofline = np.minimum(memory_roof, compute_roof)

    plt.plot(intensity_range, roofline, color='red', linestyle='--', label=f'Roofline (β={BETA_PEAK:.1f}, π={PI_PEAK:.2f})')
    # Optionally plot the components
    plt.plot(intensity_range, memory_roof, color='grey', linestyle=':', label='Memory Bound')
    plt.plot(intensity_range, compute_roof, color='grey', linestyle=':', label='Compute Bound')

    # --- Formatting ---
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Operational Intensity (I) [ops/byte]')
    plt.ylabel('Performance (P) [ops/cycle]')
    plt.title(f'Roofline Plot for merge_implicants_dense')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    # Adjust limits to fit data and roofline nicely
    plt.xlim(plot_min_intensity, plot_max_intensity)
    plt.ylim(bottom=min(df['performance'].min(), PI_PEAK / 10) / 2, # Adjust bottom limit
            top=plot_max_perf)


    plt.savefig("merge_roofline.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv("measurements_merge.csv")
    print(df)
    df = df.groupby(['compiler_version', 'compiler_flags', 'cpu_model', 'implementation', 'bits']).median().reset_index()
    print(df)
    df['ops'] = 3 * df['bits'] * (2 ** df['bits'] - 1)
    df['performance'] = df['ops'] / df['cycles']
    df['transferred_bytes'] = df['bits'].apply(calculate_bytes)
    df['operational_intensity'] = df['ops'] / df['transferred_bytes']
    df['function'] = df['implementation'] + ', ' + df['compiler_version'] + ', ' + df['compiler_flags']
    print(df)
    performance_plot(df)
    #roofline_plot(df)
