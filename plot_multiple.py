import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def main():
    # Check if there are any command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot.py <csv_file1> <csv_file2> ...")
        print("No files provided. Using default 'measurements.csv'")
        csv_files = ["measurements.csv"]
    else:
        # Skip the first argument (script name)
        csv_files = sys.argv[1:]
    
    # Initialize an empty DataFrame
    df_combined = pd.DataFrame()
    
    # Read and combine all CSV files
    for file in csv_files:
        if not os.path.exists(file):
            print(f"Warning: File '{file}' not found. Skipping.")
            continue
        
        try:
            print(f"Reading {file}...")
            df_temp = pd.read_csv(file)
            # Add a column indicating the source file (optional)
            # df_temp['source_file'] = os.path.basename(file)
            df_combined = pd.concat([df_combined, df_temp], ignore_index=True)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if df_combined.empty:
        print("No valid data found in the provided files.")
        return
    
    print(f"Combined {len(csv_files)} files, total rows: {len(df_combined)}")
    
    # Group and process the data
    df = df_combined.groupby(['compiler_version', 'compiler_flags', 'cpu_model', 'implementation', 'bits']).median().reset_index()
    df['ops'] = df['bits'] * 3**df['bits']
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
        
        # Generate unique filename based on number of input files
        if len(csv_files) > 1:
            filename = f"performance_combined_{len(csv_files)}_files.png"
        else:
            filename = "performance.png"
        
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    main()