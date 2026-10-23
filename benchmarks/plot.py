import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

def generate_research_plots():
    # -------------------------------------------------------------------------
    # 1. Load and Verify Datasets
    # -------------------------------------------------------------------------
    old_file = "vrp_old_multiseed_benchmark.csv"
    v3_file = "vrp_v3_multiseed_benchmark.csv"

    if not os.path.exists(old_file) or not os.path.exists(v3_file):
        raise FileNotFoundError(
            f"Missing required benchmark CSV files. Ensure both '{old_file}' "
            f"and '{v3_file}' exist in the execution directory."
        )

    df_old = pd.read_csv(old_file)
    df_v3 = pd.read_csv(v3_file)

    # Compute Total Latency = Matrix Setup + Solver Iterations
    df_old["Total_Time_s"] = df_old["Matrix_Time_s"] + df_old["Solver_Time_s"]
    df_v3["Total_Time_s"] = df_v3["Matrix_Time_s"] + df_v3["Solver_Time_s"]

    # Normalize algorithm keys for side-by-side mapping
    df_old["Algo_Key"] = df_old["Algorithm"].replace({"Dijkstra": "Constructive"})
    df_v3["Algo_Key"] = df_v3["Algorithm"].replace({"A*": "Constructive"})

    # -------------------------------------------------------------------------
    # 2. Aggregate Mean and Standard Deviation Across Seeds
    # -------------------------------------------------------------------------
    agg_old = (
        df_old.groupby(["N", "Algo_Key"])
        .agg(
            Score_Mean=("Score", "mean"),
            Score_Std=("Score", "std"),
            Time_Mean=("Total_Time_s", "mean"),
            Time_Std=("Total_Time_s", "std"),
        )
        .reset_index()
    )

    agg_v3 = (
        df_v3.groupby(["N", "Algo_Key"])
        .agg(
            Score_Mean=("Score", "mean"),
            Score_Std=("Score", "std"),
            Time_Mean=("Total_Time_s", "mean"),
            Time_Std=("Total_Time_s", "std"),
        )
        .reset_index()
    )

    scales = sorted(df_v3["N"].unique())

    # -------------------------------------------------------------------------
    # 3. Configure Publication Style Parameters
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    (ax1, ax2), (ax3, ax4) = axes

    # Color Palette Definitions
    c_qpso_v3 = "#1f77b4"      # Deep Blue
    c_qpso_old = "#aec7e8"     # Soft Blue
    c_ga_v3 = "#ff7f0e"        # Deep Orange
    c_ga_old = "#ffbb78"       # Soft Orange
    c_constr_v3 = "#2ca02c"    # Deep Green
    c_constr_old = "#98df8a"   # Soft Green

    # -------------------------------------------------------------------------
    # PANEL A: Solution Score Quality vs. Scale (Log Scale + Error Band)
    # -------------------------------------------------------------------------
    for algo, c_old, c_v3, marker in [
        ("Constructive", c_constr_old, c_constr_v3, "^"),
        ("GA", c_ga_old, c_ga_v3, "s"),
        ("QPSO", c_qpso_old, c_qpso_v3, "o"),
    ]:
        # Old Baseline
        sub_o = agg_old[agg_old["Algo_Key"] == algo].sort_values("N")
        label_o = f"{algo} (Legacy 1D Key)" if algo != "Constructive" else "Dijkstra (Legacy)"
        ax1.plot(sub_o["N"], sub_o["Score_Mean"], label=label_o, color=c_old, linestyle="--", marker=marker, alpha=0.85)
        ax1.fill_between(sub_o["N"], sub_o["Score_Mean"] - sub_o["Score_Std"].fillna(0), sub_o["Score_Mean"] + sub_o["Score_Std"].fillna(0), color=c_old, alpha=0.15)

        # New V3 API
        sub_n = agg_v3[agg_v3["Algo_Key"] == algo].sort_values("N")
        label_n = f"{algo} (V3 Neural Hybrid)" if algo != "Constructive" else "Neural A* (V3 API)"
        ax1.plot(sub_n["N"], sub_n["Score_Mean"], label=label_n, color=c_v3, linestyle="-", marker=marker, linewidth=2.2)
        ax1.fill_between(sub_n["N"], sub_n["Score_Mean"] - sub_n["Score_Std"].fillna(0), sub_n["Score_Mean"] + sub_n["Score_Std"].fillna(0), color=c_v3, alpha=0.20)

    ax1.set_yscale("log")
    ax1.set_xlabel("Customer Count (N)", fontweight="bold")
    ax1.set_ylabel("Route Objective Score (Log Scale)", fontweight="bold")
    ax1.set_title("(a) Route Solution Quality vs. Scale (Mean ± 1σ)", fontweight="bold", loc="left")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    # -------------------------------------------------------------------------
    # PANEL B: Total Latency Scaling (Log Scale: ms to s)
    # -------------------------------------------------------------------------
    for algo, c_old, c_v3, marker in [
        ("Constructive", c_constr_old, c_constr_v3, "^"),
        ("GA", c_ga_old, c_ga_v3, "s"),
        ("QPSO", c_qpso_old, c_qpso_v3, "o"),
    ]:
        sub_o = agg_old[agg_old["Algo_Key"] == algo].sort_values("N")
        sub_n = agg_v3[agg_v3["Algo_Key"] == algo].sort_values("N")

        ax2.plot(sub_o["N"], sub_o["Time_Mean"], color=c_old, linestyle="--", marker=marker, alpha=0.85)
        ax2.plot(sub_n["N"], sub_n["Time_Mean"], color=c_v3, linestyle="-", marker=marker, linewidth=2.0)

    ax2.set_yscale("log")
    ax2.set_xlabel("Customer Count (N)", fontweight="bold")
    ax2.set_ylabel("Total Time (Seconds, Log Scale)", fontweight="bold")
    ax2.set_title("(b) Computation Latency Scaling (ms to s)", fontweight="bold", loc="left")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # PANEL C: Efficiency Frontier Scatter at N = 500
    # -------------------------------------------------------------------------
    n_focus = 500
    sub_old_500 = agg_old[agg_old["N"] == n_focus]
    sub_v3_500 = agg_v3[agg_v3["N"] == n_focus]

    for _, row in sub_old_500.iterrows():
        ax3.errorbar(
            row["Time_Mean"], row["Score_Mean"],
            xerr=row["Time_Std"], yerr=row["Score_Std"],
            fmt="o", color="#d62728", ecolor="#d62728", elinewidth=1.5, capsize=4, markersize=8, zorder=5
        )
        ax3.annotate(
            f"Old {row['Algo_Key']}",
            (row["Time_Mean"] * 1.08, row["Score_Mean"]),
            fontsize=8, color="#8c1111", fontweight="bold"
        )

    for _, row in sub_v3_500.iterrows():
        ax3.errorbar(
            row["Time_Mean"], row["Score_Mean"],
            xerr=row["Time_Std"], yerr=row["Score_Std"],
            fmt="s", color="#2ca02c", ecolor="#2ca02c", elinewidth=1.5, capsize=4, markersize=8, zorder=5
        )
        ax3.annotate(
            f"V3 {row['Algo_Key']}",
            (row["Time_Mean"] * 1.08, row["Score_Mean"]),
            fontsize=8, color="#1b5e20", fontweight="bold"
        )

    ax3.set_yscale("log")
    ax3.set_xscale("log")
    ax3.set_xlabel("Total Execution Time (Seconds, Log Scale)", fontweight="bold")
    ax3.set_ylabel("Route Objective Score (Log Scale)", fontweight="bold")
    ax3.set_title(f"(c) Time vs. Performance Frontier (N={n_focus})", fontweight="bold", loc="left")
    ax3.grid(True, which="both", linestyle=":", alpha=0.6)
    
    # Highlight Pareto Feasibility Zone
    ax3.axhspan(700000, 850000, color="green", alpha=0.08, label="V3 Optimal Zone")
    ax3.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # PANEL D: Relative Score Reduction (%) of V3 vs. Legacy
    # -------------------------------------------------------------------------
    qpso_old_m = agg_old[agg_old["Algo_Key"] == "QPSO"].sort_values("N")["Score_Mean"].values
    qpso_v3_m = agg_v3[agg_v3["Algo_Key"] == "QPSO"].sort_values("N")["Score_Mean"].values
    
    ga_old_m = agg_old[agg_old["Algo_Key"] == "GA"].sort_values("N")["Score_Mean"].values
    ga_v3_m = agg_v3[agg_v3["Algo_Key"] == "GA"].sort_values("N")["Score_Mean"].values

    qpso_imp = ((qpso_old_m - qpso_v3_m) / qpso_old_m) * 100.0
    ga_imp = ((ga_old_m - ga_v3_m) / ga_old_m) * 100.0

    x = np.arange(len(scales))
    width = 0.35

    ax4.bar(x - width / 2, ga_imp, width, label="GA Improvement (%)", color=c_ga_v3, edgecolor="black", alpha=0.85)
    ax4.bar(x + width / 2, qpso_imp, width, label="QPSO Improvement (%)", color=c_qpso_v3, edgecolor="black", alpha=0.85)

    ax4.axhline(0, color="black", linewidth=1.0)
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(s) for s in scales])
    ax4.set_xlabel("Customer Count (N)", fontweight="bold")
    ax4.set_ylabel("Cost Reduction Over Legacy API (%)", fontweight="bold")
    ax4.set_title("(d) V3 Relative Score Improvement (%)", fontweight="bold", loc="left")
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left")

    # Annotate key percentages on Panel D
    for i, (g_v, q_v) in enumerate(zip(ga_imp, qpso_imp)):
        if q_v > 0:
            ax4.text(i + width / 2, q_v + 1.5, f"{q_v:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # -------------------------------------------------------------------------
    # 4. Final Formatting and File Export
    # -------------------------------------------------------------------------
    plt.tight_layout()
    
    png_out = "research_benchmark_comparison.png"
    pdf_out = "research_benchmark_comparison.pdf"
    
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    print(f"✅ Publication plots successfully rendered:")
    print(f"   • Raster Graphic: {png_out} (300 DPI)")
    print(f"   • Vector Graphic: {pdf_out} (Publication ready)")

if __name__ == "__main__":
    generate_research_plots()