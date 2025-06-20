import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def performance_plot(df):
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

def roofline_plot(df):
    # TODO: use hw values here
    PI_PEAK = 3 * 256.0 # Skylake: throughput of 3 for _mm256_and_si256 -> 3*356 logic operations per second
    BETA_PEAK = 30.0

    cpu_model = df['cpu_model'][0]

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
    plt.title(f"Roofline plot all implementations for different number of bits n\nCPU: {cpu_model}", loc='left')
    plt.xlabel('Operational Intensity (I) [ops/byte]')
    plt.ylabel('Performance (P) [ops/cycle]')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    # Adjust limits to fit data and roofline nicely
    plt.xlim(plot_min_intensity, plot_max_intensity)
    plt.ylim(bottom=min(df['performance'].min(), PI_PEAK / 10) / 2, # Adjust bottom limit
            top=plot_max_perf)


    plt.savefig("merge_roofline.png", dpi=300, bbox_inches="tight")
    plt.show()

def runtime_plot(df):
    """
    Generates a log-runtime plot for different implementations.
    Creates a separate plot for each CPU model found in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'bits', 'cycles',
                           'function', and 'cpu_model' columns.
    """
    # Group by CPU model to create a separate plot for each
    for i, df_grouped in df.groupby(['cpu_model']):
        # Get the CPU model name for the title and filename
        # reset_index is not strictly necessary here but can prevent potential warnings
        df_reset = df_grouped.reset_index()
        cpu_model = df_reset['cpu_model'][0]

        # Create the plot
        plt.figure(figsize=(9, 6))
        sns.lineplot(data=df_grouped, x='bits', y='cycles', hue='function', marker='o')

        # --- Formatting ---
        # Set y-axis to a logarithmic scale, which is crucial for runtime plots
        plt.yscale('log')

        # Labels and Title
        plt.xlabel('n (Input Bits)')
        plt.ylabel('Runtime [cycles] (log scale)')
        plt.title(f"Runtime of all implementations for different number of bits n\nCPU: {cpu_model}", loc='left')

        # Grid and Ticks for better readability
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.xticks(df_grouped['bits'].unique())

        # Save and show the plot
        # Create a unique filename for each CPU to avoid overwriting
        safe_cpu_model_name = cpu_model.replace(' ', '_').replace('@', '')
        plt.show()


def speedup_plot(df):
    """
    For each (cpu_model, n), compute speedup of each implementation relative
    to the 'bits' implementation at that same n. Then plot speedup vs. n.
    """
    # We assume there is a column 'implementation' and one of them is exactly 'bits'.
    for cpu_model, df_cpu in df.groupby('cpu_model'):
        # pivot so that we can easily look up the 'bits' cycles for each n
        bits_cycles = (
            df_cpu[df_cpu['implementation'] == 'bits']
            .set_index('bits')['cycles']
        )
        # some sanity check
        if bits_cycles.empty:
            print(f"Warning: no 'bits' implementation found for CPU={cpu_model}")
            continue

        # compute speedup column
        df_cpu = df_cpu.copy()
        # map each row’s n → the baseline cycles, then divide
        df_cpu['baseline_cycles'] = df_cpu['bits'].map(bits_cycles)
        df_cpu['speedup'] = df_cpu['baseline_cycles'] / df_cpu['cycles']

        # drop any rows where baseline is missing
        df_cpu = df_cpu.dropna(subset=['baseline_cycles'])

        plt.figure(figsize=(9, 6))
        sns.lineplot(
            data=df_cpu,
            x='bits',
            y='speedup',
            hue='implementation',
            marker='o'
        )

        plt.xlabel('n (input bits)')
        plt.ylabel('Speedup over “bits” impl')
        plt.title(f"Speedup vs. n (baseline = bits) CPU: {cpu_model}", loc='left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(sorted(df_cpu['bits'].unique()))
        plt.ylim(bottom=0)

        plt.legend(title='Implementation', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

def cache_plot(df):
    """
    Generates a cache miss rate plot for different implementations.
    Creates a separate plot for each CPU model found in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'bits', 'cache_miss_rate',
                           'function', and 'cpu_model' columns.
    """
    # Group by CPU model to create a separate plot for each
    for i, df_grouped in df.groupby(['cpu_model']):
        # Get the CPU model name for the title and filename
        df_reset = df_grouped.reset_index()
        cpu_model = df_reset['cpu_model'][0]

        # Create the plot
        plt.figure(figsize=(9, 6))
        sns.lineplot(data=df_grouped, x='bits', y='cache_miss_rate', hue='function', marker='o')

        # --- Formatting ---
        plt.xlabel('n (Input Bits)')
        plt.ylabel('Cache Miss Rate')
        plt.title(f"Cache Miss Rate of all implementations for different number of bits n\nCPU: {cpu_model}", loc='left')
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.xticks(df_grouped['bits'].unique())

        # Save and show the plot
        plt.show()




df = pd.read_csv("measurements.csv")
df = df.groupby(['compiler_version', 'compiler_flags', 'cpu_model', 'implementation', 'bits']).median().reset_index()
df['ops'] = df['bits'] * 3**df['bits']
df['memory'] = (3**df['bits']) / 4
df['operational_intensity'] = df['ops'] / df['memory']
df['performance'] = df['ops'] / df['cycles']
df['function'] = df['implementation'] + ', ' + df['compiler_version'] + ', ' + df['compiler_flags']
df['cache_miss_rate'] = df['l1d_cache_misses'] / df['l1d_cache_accesses']
print(df)
performance_plot(df)
roofline_plot(df)
runtime_plot(df)
speedup_plot(df)
cache_plot(df)
