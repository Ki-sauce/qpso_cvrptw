import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_benchmark_results(csv_filename="benchmark_combined.csv"):
    if not os.path.exists(csv_filename):
        # Fallback to alternative naming convention if necessary
        fallback = "vrp_unified_benchmark_results.csv"
        if os.path.exists(fallback):
            csv_filename = fallback
        else:
            print(f"❌ Error: Could not find '{csv_filename}' or '{fallback}'. Run your benchmark script first!")
            return

    print(f"📊 Loading benchmark data from: {csv_filename}")
    df = pd.read_csv(csv_filename)

    # Ensure required columns exist
    required_cols = {"N", "Algorithm", "Mean_Cost", "Mean_Exec_s"}
    if not required_cols.issubset(df.columns):
        print(f"❌ Error: CSV missing required columns. Found: {df.columns.tolist()}")
        return

    # Aggregate data by N and Algorithm (in case multiple weights/capacities are present)
    agg_df = df.groupby(["N", "Algorithm"]).agg({
        "Mean_Cost": "mean",
        "Mean_Exec_s": "mean"
    }).reset_index()

    # Set up a modern plot style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color mapping for algorithms
    colors = {"A*": "#e74c3c", "GA": "#3498db", "QPSO": "#2ecc71"}

    # --- PLOT 1: SOLUTION QUALITY (Cost vs N) ---
    for algo, group in agg_df.groupby("Algorithm"):
        ax1.plot(
            group["N"], group["Mean_Cost"], 
            marker='o', linewidth=2.5, markersize=7, 
            label=algo, color=colors.get(algo, '#333333')
        )
    
    ax1.set_title("Solution Quality Scaling ($N = 10$ to $90$)", fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel("Number of Customers ($N$)", fontsize=11)
    ax1.set_ylabel("Mean Cost Score (Lower is Better)", fontsize=11)
    ax1.legend(title="Algorithm", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- PLOT 2: COMPUTATIONAL SPEED (Time vs N) ---
    for algo, group in agg_df.groupby("Algorithm"):
        ax2.plot(
            group["N"], group["Mean_Exec_s"], 
            marker='s', linewidth=2.5, markersize=7, 
            label=algo, color=colors.get(algo, '#333333')
        )

    ax2.set_title("Execution Speed vs. Scale ($N = 10$ to $90$)", fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel("Number of Customers ($N$)", fontsize=11)
    ax2.set_ylabel("Mean Execution Time (Seconds)", fontsize=11)
    ax2.legend(title="Algorithm", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    output_image = "vrp_benchmark_performance_curves.png"
    plt.savefig(output_image, dpi=300)
    print(f"📈 Chart successfully saved to: {output_image}")
    
    plt.show()

if __name__ == "__main__":
    plot_benchmark_results()