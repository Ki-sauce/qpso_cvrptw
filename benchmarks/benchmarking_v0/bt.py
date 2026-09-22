import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_advanced_performance(csv_filename="vrp_throttled_results.csv"):
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Could not locate {csv_filename}. Please run the benchmark script first.")
        return

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    # 1. Main Cost Comparison (QPSO vs GA)
    ax1 = fig.add_subplot(gs[0, :])
    meta_df = df[df["Algorithm"].isin(["QPSO", "GA"])]
    
    sns.barplot(
        data=meta_df,
        x="N",
        y="Mean_Cost",
        hue="Algorithm",
        palette={"QPSO": "#D90429", "GA": "#F4A261"},
        ax=ax1,
        errorbar="sd",
        capsize=0.1,
        err_kws={'linewidth': 1.5}
    )
    ax1.set_title("Metaheuristic Convergence: QPSO vs. Genetic Algorithm (Mean Cost & Standard Deviation)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Delivery Stops (N)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Global Objective Cost (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.legend(title="Optimization Engine", frameon=True, facecolor='white')

    # 2. Percentage Improvement of QPSO over GA
    ax2 = fig.add_subplot(gs[1, 0])
    pivot_cost = df.pivot_table(index=["N", "Weight", "Capacity", "Vehicles"], columns="Algorithm", values="Mean_Cost").reset_index()
    
    if "QPSO" in pivot_cost.columns and "GA" in pivot_cost.columns:
        pivot_cost["QPSO_Improvement_%"] = ((pivot_cost["GA"] - pivot_cost["QPSO"]) / pivot_cost["GA"]) * 100
        
        sns.barplot(
            data=pivot_cost,
            x="N",
            y="QPSO_Improvement_%",
            hue="Weight",
            palette="crest",
            ax=ax2,
            errorbar="sd",
            capsize=0.08
        )
        ax2.set_title("QPSO Cost Reduction Advantage over GA (%)", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Delivery Stops (N)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Cost Reduction Advantage (%)", fontsize=11, fontweight="bold")
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.legend(title="Distance Weight (Beta)", frameon=True)

    # 3. Execution Latency Profile (Corrected ax parameter)
    ax3 = fig.add_subplot(gs[1, 1])
    sns.lineplot(
        data=meta_df,
        x="N",
        y="Mean_Time",
        hue="Algorithm",
        style="Algorithm",
        palette={"QPSO": "#D90429", "GA": "#F4A261"},
        markers=True,
        dashes=False,
        linewidth=2.5,
        ax=ax3
    )
    ax3.set_title("Computational Scalability & Execution Latency", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("Delivery Stops (N)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
    ax3.legend(title="Algorithm", frameon=True)

    fig.suptitle("Advanced Performance Analytics: Demonstrating QPSO Superiority", fontsize=16, fontweight="bold", y=0.96)
    
    output_image = "vrp_advanced_performance_analysis.png"
    fig.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"Advanced visualization successfully generated and saved to: {output_image}")
    plt.show()

if __name__ == "__main__":
    visualize_advanced_performance()