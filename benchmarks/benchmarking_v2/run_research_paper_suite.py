import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import concurrent.futures
from scipy.stats import wilcoxon

from core.graph_model import load_osm_network
from core.traffic import apply_traffic_scenario
from core.vrp import Customer, VRPInstance
from core.engine import solve_ga_baseline, solve_qpso

def run_experiment_task(task_params):
    """
    Executes a clean comparative trial between GA and QPSO
    for statistical significance testing.
    """
    G0, n, seed, algorithm_name, dist_weight, vehicles, capacity, particles, iterations = task_params
    
    try:
        G = apply_traffic_scenario(G0, seed=seed, mode='simulated')
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'distance_m' not in data: data['distance_m'] = float(data.get('length', 10.0))
            if 'travel_time_s' not in data: data['travel_time_s'] = data['distance_m'] / 8.33

        rng = np.random.default_rng(seed)
        components = nx.strongly_connected_components(G) if G.is_directed() else nx.connected_components(G)
        nodes = np.asarray(list(max(components, key=len)))
        chosen = rng.choice(nodes, size=n + 1, replace=False)
        depot, customer_nodes = int(chosen[0]), [int(x) for x in chosen[1:]]
        
        demands = rng.integers(1, 6, size=len(customer_nodes))
        customers_obj = [
            Customer(
                node=node,
                demand=float(d),
                ready_time=32400.0 + float(rng.uniform(0, 3600)),
                due_time=32400.0 + float(rng.uniform(0, 3600)) + 10800.0,
                service_time=300.0,
            )
            for node, d in zip(customer_nodes, demands)
        ]
        
        instance = VRPInstance(depot=depot, customers=customers_obj, vehicle_capacity=capacity, num_vehicles=vehicles)

        start_time = time.perf_counter()
        if algorithm_name == "GA-Baseline":
            res = solve_ga_baseline(G, instance, particles=particles, iterations=iterations, seed=seed, distance_weight=dist_weight)
        else:
            res = solve_qpso(G, instance, particles=particles, iterations=iterations, seed=seed, distance_weight=dist_weight)
        exec_time = time.perf_counter() - start_time
        
        return {
            "N": n,
            "Seed": seed,
            "Algorithm": algorithm_name,
            "Cost": res.get('score', float('inf')),
            "Distance_km": res.get('distance_m', float('inf')) / 1000.0,
            "Time_s": exec_time
        }
    except Exception as exc:
        print(f"⚠️ Task failed ({algorithm_name}, seed={seed}): {exc}")
        return None

def run_research_suite():
    LOCATION = "Manhattan, New York, USA"
    
    N = 35                                    # Goldilocks scale
    VEHICLES = 4
    CAPACITY = 50.0
    DIST_WEIGHT = 0.50
    TRIALS = 30                               # Statistically robust sample size
    PARTICLES = 35
    ITERATIONS = 80
    
    ALGORITHMS = ["GA-Baseline", "QPSO-Engine"]

    max_workers = os.cpu_count()
    print(f"🔬 Launching Publication-Grade Comparative Suite across {max_workers} M1 cores...")
    print(f"📊 Collecting {TRIALS} independent trial pairs for Wilcoxon statistical validation.\n")

    G0 = load_osm_network(LOCATION)

    tasks = []
    for algo in ALGORITHMS:
        for trial in range(TRIALS):
            seed = 100 + trial
            tasks.append((G0, N, seed, algo, DIST_WEIGHT, VEHICLES, CAPACITY, PARTICLES, ITERATIONS))

    all_results = []
    start_bench = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_experiment_task, t) for t in tasks]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res: all_results.append(res)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"   [{i + 1}/{len(tasks)}] Evaluation runs completed...")

    print(f"\n✅ Research sweep concluded in {time.perf_counter() - start_bench:.2f} seconds.")

    df = pd.DataFrame(all_results)
    if df.empty:
        raise RuntimeError("Research suite failed to capture valid runs.")

    # --- AGGREGATION & SUMMARY ---
    summary = df.groupby("Algorithm").agg(
        Mean_Cost=("Cost", "mean"),
        Std_Cost=("Cost", "std"),
        Median_Cost=("Cost", "median"),
        Mean_Dist_km=("Distance_km", "mean"),
        Mean_Exec_s=("Time_s", "mean")
    ).reset_index().sort_values(by="Mean_Cost")

    print("\n🏆 Table 1: Comparative Performance Breakdown ($N=35$):")
    print("========================================================================================")
    print(summary.to_string(index=False))
    print("========================================================================================")

    # --- STATISTICAL SIGNIFICANCE TESTING (Wilcoxon Signed-Rank Test) ---
    print("\n📈 Table 2: Statistical Significance Analysis (Wilcoxon Signed-Rank Test):")
    print("----------------------------------------------------------------------------------------")
    
    ga_costs = df[df["Algorithm"] == "GA-Baseline"].sort_values("Seed")["Cost"].values
    qpso_costs = df[df["Algorithm"] == "QPSO-Engine"].sort_values("Seed")["Cost"].values
    
    if len(ga_costs) == len(qpso_costs) and len(ga_costs) > 0:
        stat, p_value = wilcoxon(ga_costs, qpso_costs)
        sig_df = pd.DataFrame([{
            "Comparison": "GA-Baseline vs QPSO-Engine",
            "Wilcoxon Statistic": stat,
            "p-value": p_value,
            "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
        }])
        pd.set_option('display.float_format', lambda x: '%.6f' % x if x < 0.01 else '%.2f' % x)
        print(sig_df.to_string(index=False))
        sig_df.to_csv("paper_table2_significance.csv", index=False)

    summary.to_csv("paper_table1_performance.csv", index=False)
    print("\n📁 Formatted tables exported as CSV ready for LaTeX inclusion (`paper_table1_performance.csv`, `paper_table2_significance.csv`).")

if __name__ == "__main__":
    run_research_suite()