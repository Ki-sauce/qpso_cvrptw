import os
import time
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import concurrent.futures

from core.graph_model import load_osm_network
from core.traffic import apply_traffic_scenario
from core.vrp import Customer, VRPInstance
from core.engine import solve_qpso, solve_ga_baseline
from core.heuristics import solve_dynamic_heuristic

def choose_stops(G, n, seed):
    """Selects a connected depot and n customers deterministically."""
    rng = np.random.default_rng(seed)
    components = nx.strongly_connected_components(G) if G.is_directed() else nx.connected_components(G)
    largest = max(components, key=len)
    nodes = np.asarray(list(largest))
    
    if len(nodes) < n + 1:
        raise ValueError(f"Network requires {n + 1} connected nodes.")
    
    chosen = rng.choice(nodes, size=n + 1, replace=False)
    return int(chosen[0]), [int(x) for x in chosen[1:]]

def run_experiment_task(task_params):
    """
    Isolated worker function designed for multi-core parallel execution.
    Executes all three algorithms for a single specific environment configuration.
    """
    G0, n, seed, dist_weight, vehicles, capacity, particles, iterations = task_params
    results_data = []
    
    try:
        # 1. Environment Instantiation
        G = apply_traffic_scenario(G0, seed=seed, mode="simulated")
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'distance_m' not in data: data['distance_m'] = float(data.get('length', 10.0))
            if 'travel_time_s' not in data: data['travel_time_s'] = data['distance_m'] / 8.33

        depot, customer_nodes = choose_stops(G, n, seed=seed)
        rng = np.random.default_rng(seed)
        demands = rng.integers(1, 8, size=len(customer_nodes))
        customers_obj = [
            Customer(
                node=node,
                demand=float(d),
                ready_time=32400.0 + float(rng.uniform(0, 7200)),
                due_time=32400.0 + float(rng.uniform(0, 7200)) + 14400.0,
                service_time=300.0,
            )
            for node, d in zip(customer_nodes, demands)
        ]
        
        instance = VRPInstance(depot=depot, customers=customers_obj, vehicle_capacity=capacity, num_vehicles=vehicles)

        # 2. Algorithm Executions
        algorithms = [
            ("A* Heuristic", lambda: solve_dynamic_heuristic(G, instance, method='astar', distance_weight=dist_weight)),
            ("GA", lambda: solve_ga_baseline(G, instance, particles=particles, iterations=iterations, seed=seed, distance_weight=dist_weight)),
            ("QPSO", lambda: solve_qpso(G, instance, particles=particles, iterations=iterations, seed=seed, distance_weight=dist_weight))
        ]

        for algo_name, algo_func in algorithms:
            start_time = time.perf_counter()
            res = algo_func()
            exec_time = time.perf_counter() - start_time
            
            results_data.append({
                "Customers (N)": n,
                "Distance Weight": dist_weight,
                "Capacity": capacity,
                "Seed": seed,
                "Algorithm": algo_name,
                "Cost Score": res['score'],
                "Travel Time (s)": res.get('travel_time_s', float('inf')),
                "Distance (m)": res.get('distance_m', float('inf')),
                "Execution Time (s)": exec_time
            })
            
    except Exception as e:
        print(f"Error in task Seed={seed}, N={n}: {e}")
        
    return results_data

def run_benchmarks():
    # --- Multi-Dimensional Benchmark Configuration ---
    LOCATION = "Manhattan, New York, USA"
    
    # The Sensitivity Matrix
    CUSTOMER_SIZES = [10, 15, 20]             # Spatial Dimensionality
    DISTANCE_WEIGHTS = [0.2, 0.8]             # Time-Priority vs. Distance-Priority
    CAPACITIES = [20.0, 30.0]                 # Constrained vs. Relaxed combinatorial space
    VEHICLES = 5
    
    TRIALS_PER_COMBO = 10                     # Seeds per configuration
    PARTICLES = 40
    ITERATIONS = 100
    
    print(f"🚀 Initializing Multi-Core Benchmark Suite for {LOCATION}")
    print("⏳ Downloading base OSM graph (this happens once)...")
    G0 = load_osm_network(LOCATION)
    print(f"✅ Graph loaded: {len(G0.nodes)} nodes, {len(G0.edges)} edges.\n")

    # Generate all task combinations
    tasks = []
    for n, dist_weight, capacity in itertools.product(CUSTOMER_SIZES, DISTANCE_WEIGHTS, CAPACITIES):
        for trial in range(TRIALS_PER_COMBO):
            seed = 42 + trial
            tasks.append((G0, n, seed, dist_weight, VEHICLES, capacity, PARTICLES, ITERATIONS))

    total_tasks = len(tasks)
    print(f"⚡ Generating {total_tasks} parallel environments ({total_tasks * 3} algorithm executions)...")
    
    all_results = []
    
    # --- Parallel Execution Engine ---
    start_bench = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_experiment_task, task): task for task in tasks}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            task_res = future.result()
            if task_res:
                all_results.extend(task_res)
            completed += 1
            if completed % 10 == 0 or completed == total_tasks:
                print(f"   [{completed}/{total_tasks}] Environments evaluated...")

    total_time = time.perf_counter() - start_bench
    print(f"\n✅ All parallel executions finished in {total_time:.2f} seconds.")

    # --- Data Cleaning & Aggregation ---
    df = pd.DataFrame(all_results)
    if df.empty:
        raise RuntimeError("No benchmark results were produced; inspect task errors above.")
    valid_df = df[df["Cost Score"] < 900000].copy()
    if valid_df.empty:
        raise RuntimeError("All benchmark runs were infeasible or invalid.")

    summary_df = valid_df.groupby(["Customers (N)", "Distance Weight", "Capacity", "Algorithm"]).agg(
        Mean_Cost=("Cost Score", "mean"),
        Std_Cost=("Cost Score", "std"),
        Min_Cost=("Cost Score", "min"),
        Mean_Time_s=("Execution Time (s)", "mean")
    ).reset_index()

    success_counts = valid_df.groupby(["Customers (N)", "Distance Weight", "Capacity", "Algorithm"]).size().reset_index(name='Valid_Runs')
    summary_df = pd.merge(summary_df, success_counts, on=["Customers (N)", "Distance Weight", "Capacity", "Algorithm"])
    summary_df["Success_Rate_%"] = (summary_df["Valid_Runs"] / TRIALS_PER_COMBO) * 100
    summary_df = summary_df.drop(columns=['Valid_Runs'])

    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print("\n🏆 Macroscopic Sensitivity Analysis:")
    print(summary_df.to_string(index=False))

    output_filename = "vrp_benchmark_sensitivity_matrix.csv"
    summary_df.to_csv(output_filename, index=False)
    print(f"\n📁 Data successfully exported to: {output_filename}")

if __name__ == "__main__":
    run_benchmarks()
