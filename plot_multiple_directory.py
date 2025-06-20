import pandas as pd
import sys
import os
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_random_color(existing_colors):
    # Use matplotlib's tab20 palette for distinct colors, fallback to random if exhausted
    tab_colors = list(plt.get_cmap('tab20').colors)
    while True:
        if tab_colors:
            color = tab_colors.pop(0)
        else:
            color = (random.random(), random.random(), random.random())
        if color not in existing_colors:
            return color


def read_all_csvs_with_random_colors(directory):
    """
    Reads all CSV files in the given directory, assigns each a random color, and merges them.
    Returns a DataFrame with an added 'color' column.
    """
    df_combined = pd.DataFrame()
    existing_colors = set()
    for fname in sorted(os.listdir(directory)):
        if fname.endswith('.csv'):
            file_path = os.path.join(directory, fname)
            print(f"Processing file: {file_path}")
            try:
                print(f"Reading {file_path} ...")
                df_temp = pd.read_csv(file_path)
                color = get_random_color(existing_colors)
                existing_colors.add(color)
                df_temp['color'] = [color] * len(df_temp)
                print(df_temp.head())  # Print first few rows for debugging
                df_combined = pd.concat([df_combined, df_temp], ignore_index=True)
                print(df_combined.head())  # Print first few rows for debugging
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return df_combined


def parse_args_with_colors():
    """
    Parse command line arguments to get file-color pairs.
    Format: <file1> <color1> <file2> <color2> ...
    Returns a list of (file, color) tuples.
    """
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("Usage: python plot_2.py <csv_file1> <color1> <csv_file2> <color2> ...")
        print("Example: python plot_2.py data1.csv blue data2.csv red")
        exit(1)
    
    # Skip the script name and group into pairs
    args = sys.argv[1:]
    file_color_pairs = []
    for i in range(0, len(args), 2):
        if i + 1 < len(args):
            file_color_pairs.append((args[i], args[i + 1]))
    
    return file_color_pairs


def read_files_with_colors(file_color_pairs):
    """
    Read multiple CSV files and assign colors based on the provided color pairs.
    Returns a DataFrame with an added 'color' column.
    """
    df_combined = pd.DataFrame()
    
    for file_path, color in file_color_pairs:
        if not os.path.exists(file_path):
            print(f"Warning: File '{file_path}' not found. Skipping.")
            continue
        
        try:
            print(f"Reading {file_path} with color {color}...")
            df_temp = pd.read_csv(file_path)
            print(df_temp.head())  # Print first few rows for debugging
            # Add source file and color columns
            # df_temp['source_file'] = os.path.basename(file_path)
            df_temp['color'] = [color] * len(df_temp)
            df_combined = pd.concat([df_combined, df_temp], ignore_index=True)
            print(df_combined.head())  # Print first few rows for debugging
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return df_combined


def performance_plot(df):
    for i, df_grouped in df.groupby(['cpu_model']):
        df_reset = df_grouped.reset_index()
        cpu_model = df_reset['cpu_model'][0]
        compiler_version = df_reset['compiler_version'][0] + " " + df_reset['compiler_flags'][0]
        plt.figure(figsize=(14, 6))  # Wider figure for legend space
        handles = []
        labels = []
        min_perf = df_grouped['performance'].min()
        max_perf = df_grouped['performance'].max()
        # Group by function and color for custom coloring
        for (func, color), group in df_grouped.groupby(['function', 'color']):
            group = group.sort_values('bits')
            compiler_info = group['compiler_version'].iloc[0] + " " + group['compiler_flags'].iloc[0]
            label = f"{func} [{compiler_info}]"
            handle, = plt.plot(group['bits'], group['performance'], 
                               marker='o', linestyle='-', color=color, label=label)
            handles.append(handle)
            labels.append(label)
            last_point = group.iloc[-1]
            text_x = last_point['bits']
            text_y = last_point['performance']
            halign = 'left'
            offset_x = 0.2
            if text_x == df_grouped['bits'].max():
                halign = 'right'
                offset_x = 0.2
            plt.annotate(func,
                        xy=(text_x, text_y),
                        xytext=(text_x + offset_x, text_y * (1.03 + int(func == 'hellman') * 0.05)),
                        color=color,
                        fontsize=9,
                        fontweight='bold',
                        horizontalalignment=halign,
                        verticalalignment='center')

        plt.xlabel('n')
        plt.ylabel('Performance [ops/cycle]')
        plt.title(f"Performance \nCPU: {cpu_model}\nCompiler: {compiler_version}", loc='left')
        plt.grid(True)
        plt.autoscale(enable=True, axis='y', tight=False)
        # Set y-limits to zoom in: start just below the minimum value
        plt.ylim(bottom=max(0, min_perf * 0.95), top=max_perf * 1.05)
        plt.xticks(df_grouped['bits'].unique())
        plt.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc='upper left', title='Implementation [Compiler]', borderaxespad=0.)
        plt.subplots_adjust(right=0.65)  # Leave space on the right for the legend
        plt.tight_layout()
        plt.savefig("performance.png", dpi=300, bbox_inches="tight")
        plt.show()

def roofline_plot(df):
    # TODO: use hw values here
    PI_PEAK = 3 * 512.0 # Zen 4: throughput of 3 for _mm512_and_si512 -> 3*512 logic operations per second
    
    # Memory bandwidth peak in B/cycle
    # Obtained by averaging 50 custom STREAM runs (array size increased to 2 mil elements) 
    BETA_PEAK = 9.06 # 29899.75 MB/s / 3300 MHz = 9.06 B/cycle (for Zen 4)

    # Add your custom rooflines
    PI_PEAK_64 = 4 * 64 * 2 
    PI_PEAK_256 = 4 * 256 * 2 
    PI_PEAK_512 = 4 * 512 * 2

    cpu_model = df['cpu_model'][0]
    compiler_version = df['compiler_version'][0] + " " + df['compiler_flags'][0]
    
    plt.figure(figsize=(10, 7))
    
    # Plot the measured performance points with custom colors
    for (function, color), group in df.groupby(['function', 'color']):
        # Sort by operational_intensity to ensure proper plotting
        group = group.sort_values('operational_intensity')
        
        # Plot this function's line with the specified color
        plt.plot(group['operational_intensity'], group['performance'], 
                 marker='o', linestyle='-', color=color, label=function)
        
        # Annotate first point with the bit value
        first_point = group.iloc[0]
        plt.text(first_point['operational_intensity'], 
                first_point['performance'],
                f' n={first_point["bits"]}',
                fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                color=color)
        
        # Annotate last point with bit value and function name
        last_point = group.iloc[-1]
        
        # Place function name at the end of the line in the same color
        plt.text(last_point['operational_intensity'] * 1.05, 
                last_point['performance'] * 1.05,
                f"{function}",
                fontsize=9, verticalalignment='center', 
                color=color, weight='bold')
                
        # Also add the bit value for the last point
        plt.text(last_point['operational_intensity'], 
                last_point['performance'] * 0.98,
                f' n={last_point["bits"]}',
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                color=color)
    
    min_intensity = df['operational_intensity'].min()
    max_intensity = df['operational_intensity'].max()
    max_perf = df['performance'].max()

    # Extend plot range slightly for visibility
    plot_min_intensity = max(min_intensity / 4, 0.1)  # Start a bit above 0
    plot_max_intensity = max_intensity * 3  # Extend more to make room for labels
    plot_max_perf = max(max_perf, PI_PEAK, PI_PEAK_64, PI_PEAK_256, PI_PEAK_512) * 1.5

    # Calculate roofline points
    # Create a range of operational intensity values for plotting the roof
    intensity_range = np.logspace(np.log2(plot_min_intensity), np.log2(plot_max_intensity), 100, base=2)

    # Memory-bound roof (Performance = Beta * Intensity)
    memory_roof = BETA_PEAK * intensity_range

    # Compute-bound roof (Performance = Pi_peak)
    compute_roof = np.full_like(intensity_range, PI_PEAK)

    # Actual roofline is the minimum of the two
    roofline = np.minimum(memory_roof, compute_roof)

    # Plot roofline and bounds with labels directly on the lines
    plt.plot(intensity_range, roofline, color='red', linestyle='--')
    plt.text(intensity_range[60], roofline[60], f' Roofline (β={BETA_PEAK:.1f}, π={PI_PEAK:.2f})', 
             color='red', fontsize=10, verticalalignment='bottom')
    
    # Optionally plot the components with direct labels
    plt.plot(intensity_range, memory_roof, color='grey', linestyle=':')
    plt.text(intensity_range[30], memory_roof[30], ' Memory Bound', 
             color='grey', fontsize=9, verticalalignment='bottom')
    
    plt.plot(intensity_range, compute_roof, color='grey', linestyle=':')
    plt.text(intensity_range[80], compute_roof[80], ' Compute Bound', 
             color='grey', fontsize=9, verticalalignment='bottom')

    # --- Add horizontal lines for custom rooflines ---
    plt.axhline(PI_PEAK_64, color='blue', linestyle='--', linewidth=1, label='π_64 = 4×64')
    plt.text(plot_min_intensity * 1.1, PI_PEAK_64, 'π_64', color='blue', va='bottom', fontsize=9)

    plt.axhline(PI_PEAK_256, color='green', linestyle='--', linewidth=1, label='π_256 = 4×256')
    plt.text(plot_min_intensity * 1.1, PI_PEAK_256, 'π_256', color='green', va='bottom', fontsize=9)

    plt.axhline(PI_PEAK_512, color='purple', linestyle='--', linewidth=1, label='π_512 = 4×512')
    plt.text(plot_min_intensity * 1.1, PI_PEAK_512, 'π_512', color='purple', va='bottom', fontsize=9)

    # --- Formatting ---
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    
    # Configure x-axis with powers of 2 ticks
    x_min_pow2 = int(np.floor(np.log2(plot_min_intensity)))
    x_max_pow2 = int(np.ceil(np.log2(plot_max_intensity)))
    x_ticks = [2**i for i in range(x_min_pow2, x_max_pow2 + 1)]
    plt.xticks(x_ticks, [f"{2**i}" for i in range(x_min_pow2, x_max_pow2 + 1)])
    
    # Set grid to align with powers of 2
    plt.grid(True, which='major', linestyle='-', linewidth=0.5)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.3)
    
    plt.title(f"Roofline plot \nCPU: {cpu_model}\nCompiler: {compiler_version}", loc='left')
    plt.xlabel('Operational Intensity (I) [ops/byte]')
    plt.ylabel('Performance (P) [ops/cycle]')
    
    # Adjust limits to fit data and roofline nicely
    plt.xlim(plot_min_intensity, plot_max_intensity)
    plt.ylim(bottom=min(df['performance'].min(), PI_PEAK / 10) / 2,  # Adjust bottom limit
             top=plot_max_perf)

    plt.tight_layout()
    plt.savefig("merge_roofline.png", dpi=300, bbox_inches="tight")
    plt.show()


def runtime_plot(df):
    """
    Generates a log-runtime plot for different implementations.
    Creates a separate plot for each CPU model found in the DataFrame.
    """
    # Group by CPU model to create a separate plot for each
    for i, df_grouped in df.groupby(['cpu_model']):
        # Get the CPU model name for the title and filename
        df_reset = df_grouped.reset_index()
        cpu_model = df_reset['cpu_model'][0]

        # Create the plot
        plt.figure(figsize=(9, 6))
        
        # Group by function and color for custom coloring
        for (func, color), group in df_grouped.groupby(['function', 'color']):
            # Sort by bits to ensure we get truly last point
            group = group.sort_values('bits')
            
            # Plot the line with custom color but no label
            plt.plot(group['bits'], group['cycles'], 
                     marker='o', linestyle='-', color=color)
            
            # Add function name at the end of the line
            last_point = group.iloc[-1]
            text_x = last_point['bits']
            text_y = last_point['cycles']
            
            # Choose horizontal alignment based on position
            halign = 'left'
            offset_x = 0.2
            if text_x == df_grouped['bits'].max():
                halign = 'right'
                offset_x = -0.2
            
            plt.annotate(func, 
                        xy=(text_x, text_y),
                        xytext=(text_x + offset_x, text_y),
                        color=color,
                        fontsize=9,
                        fontweight='bold',
                        horizontalalignment=halign,
                        verticalalignment='center')

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
        
        # Remove the legend - we're labeling lines directly
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save and show the plot
        plt.savefig("runtime.png", dpi=300, bbox_inches="tight")
        plt.show()


def speedup_plot(df):
    """
    For each (cpu_model, n), compute speedup of each implementation relative
    to the 'hellman' implementation at that same n. Then plot speedup vs. n.
    """
    # We assume there is a column 'implementation' and one of them is 'hellman'
    for cpu_model, df_cpu in df.groupby('cpu_model'):
        # pivot so that we can easily look up the 'hellman' cycles for each n
        hellman_cycles = (
            df_cpu[df_cpu['implementation'] == 'hellman']
            .set_index('bits')['cycles']
        )
        # some sanity check
        if hellman_cycles.empty:
            print(f"Warning: no 'hellman' implementation found for CPU={cpu_model}")
            continue

        # compute speedup column
        df_cpu = df_cpu.copy()
        # map each row's n → the baseline cycles, then divide
        df_cpu['baseline_cycles'] = df_cpu['bits'].map(hellman_cycles)
        df_cpu['speedup'] = df_cpu['baseline_cycles'] / df_cpu['cycles']

        # drop any rows where baseline is missing
        df_cpu = df_cpu.dropna(subset=['baseline_cycles'])

        plt.figure(figsize=(9, 6))
        
        # Group by implementation and color for custom coloring
        for (impl, color), group in df_cpu.groupby(['implementation', 'color']):
            # Sort by bits to ensure we get truly last point
            group = group.sort_values('bits')
            
            # Plot the line with custom color but no label
            plt.plot(group['bits'], group['speedup'], 
                     marker='o', linestyle='-', color=color)
            
            # Add implementation name at the end of the line
            last_point = group.iloc[-1]
            text_x = last_point['bits']
            text_y = last_point['speedup']
            
            # Choose horizontal alignment based on position
            halign = 'left'
            offset_x = 0.2
            if text_x == df_cpu['bits'].max():
                halign = 'right'
                offset_x = -0.2
            
            plt.annotate(impl, 
                        xy=(text_x, text_y),
                        xytext=(text_x + offset_x, text_y * (1 + int(impl == 'hellman') * 0.1)),
                        color=color,
                        fontsize=9,
                        fontweight='bold',
                        horizontalalignment=halign,
                        verticalalignment='center')

        plt.xlabel('n (input bits)')
        plt.ylabel('Speedup over "hellman" impl')
        plt.title(f"Speedup vs. hellman\nCPU: {cpu_model}", loc='left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(sorted(df_cpu['bits'].unique()))
        plt.ylim(bottom=0)

        # Remove the legend - we're labeling lines directly
        # plt.legend(title='Implementation', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig("speedup_vs_hellman.png", dpi=300, bbox_inches="tight")
        plt.show()


# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_multiple_random_colors.py <directory_with_csvs>")
        sys.exit(1)
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.")
        sys.exit(1)

    df_combined = read_all_csvs_with_random_colors(directory)
    
    if df_combined.empty:
        print("No valid data found in the provided files.")
        exit(1)

    print(f"Combined {len(df_combined)} rows from CSV files in {directory}")

    df = df_combined

    df = df.groupby(['compiler_version', 'compiler_flags', 'cpu_model', 'implementation', 'bits', 'color']).median().reset_index()
    df['ops'] = df['bits'] * 3**df['bits']
    df['memory'] = (3**df['bits']) / 4
    df['operational_intensity'] = df['ops'] / df['memory']
    df['performance'] = df['ops'] / df['cycles']
    df['function'] = df['implementation']
    
    df = df[(df['bits'] >= 1) & (df['bits'] <= 22)]
    
    performance_plot(df)
    # roofline_plot(df)
    # runtime_plot(df)
    # speedup_plot(df)
