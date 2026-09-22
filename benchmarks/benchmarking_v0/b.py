import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_benchmark_results(csv_filename="vrp_throttled_results.csv"):
    """
    Reads the benchmark sensitivity CSV output and generates publication-grade 
    comparative charts illustrating algorithm performance across dimensions.
    """
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filename}. Please run the benchmark script first.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # 1. Solution Cost vs. Customer Scale (N)
    sns.barplot(
        data=df, 
        x="Customers (N)", 
        y="Mean_Cost", 
        hue="Algorithm", 
        palette="muted", 
        ax=axes[0],
        errorbar=None
    )
    axes[0].set_title("Mean Objective Cost by Scale (N)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Delivery Stops (N)", fontsize=10)
    axes[0].set_ylabel("Mean Objective Cost", fontsize=10)
    axes[0].legend(title="Algorithm", frameon=True)

    # 2. Execution Time vs. Customer Scale (N)
    sns.lineplot(
        data=df, 
        x="Customers (N)", 
        y="Mean_Time_s", 
        hue="Algorithm", 
        style="Algorithm", 
        markers=True, 
        dashes=False, 
        linewidth=2.5, 
        ax=axes[1]
    )
    axes[1].set_title("Computational Latency Across Scales", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Delivery Stops (N)", fontsize=10)
    axes[1].set_ylabel("Execution Time (seconds)", fontsize=10)
    axes[1].legend(title="Algorithm", frameon=True)

    fig.suptitle("Macroscopic Sensitivity Analysis: QPSO vs. GA vs. A* Baseline", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    
    output_image = "vrp_benchmark_analysis.png"
    fig.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"Visualization successfully generated and saved to: {output_image}")
    plt.show()

if __name__ == "__main__":
    visualize_benchmark_results()