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
    nodes = np.asarray(list(max(components, key=len)))
    if len(nodes) < n + 1:
        raise ValueError(f"Network requires {n + 1} connected nodes.")
    chosen = rng.choice(nodes, size=n + 1, replace=False)
    return int(chosen[0]), [int(x) for x in chosen[1:]]

def run_experiment_task(task_params):
    """
    Isolated worker function designed for safe multi-core execution.
    Executes A*, GA, and QPSO for a single environment setup.
    """
    G0, n, seed, dist_weight, vehicles, capacity, particles, iterations = task_params
    results_data = []
    
    try:
        # 1. Environment Instantiation with Traffic
        G = apply_traffic_scenario(G0, seed=seed, mode='simulated')
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'distance_m' not in data: data['distance_m'] = float(data.get('length', 10.0))
            if 'travel_time_s' not in data: data['travel_time_s'] = data['distance_m'] / 8.33

        depot, customer_nodes = choose_stops(G, n, seed=seed)
        rng = np.random.default_rng(seed)
        demands = rng.integers(1, 6, size=len(customer_nodes))
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
        
        instance = VRPInstance(
            depot=depot, 
            customers=customers_obj, 
            vehicle_capacity=capacity, 
            num_vehicles=vehicles
        )

        # 2. Algorithm Registries
        algorithms = [
            ("A*", lambda: solve_dynamic_heuristic(G, instance, method='astar', distance_weight=dist_weight)),
            ("GA", lambda: solve_ga_baseline(G, instance, particles=particles, iterations=iterations, seed=seed, distance_weight=dist_weight)),
            ("QPSO", lambda: solve_qpso(G, instance, particles=particles, iterations=iterations, seed=seed, distance_weight=dist_weight))
        ]

        # 3. Execution & Metrics Collection
        for algo_name, algo_func in algorithms:
            start_time = time.perf_counter()
            res = algo_func()
            exec_time = time.perf_counter() - start_time
            
            results_data.append({
                "N": n,
                "Weight": dist_weight,
                "Capacity": capacity,
                "Vehicles": vehicles,
                "Seed": seed,
                "Algorithm": algo_name,
                "Cost": res.get('score', float('inf')),
                "Distance_km": res.get('distance_m', float('inf')) / 1000.0,
                "Travel_Time_min": res.get('travel_time_s', float('inf')) / 60.0,
                "Time_s": exec_time
            })
            
    except Exception as exc:
        print(f"⚠️ Task failed (N={n}, seed={seed}): {exc}")
        
    return results_data

def run_benchmarks():
    LOCATION = "Manhattan, New York, USA"
    
    # --- Benchmark Hyperparameters ---
    CUSTOMER_SIZES = [10, 20]                 # Spatial scales
    DISTANCE_WEIGHTS = [0.20, 0.80]           # Temporal vs Distance priority
    CAPACITIES = [40.0]                       # Fleet payload limits
    VEHICLE_COUNTS = [4]                      # Active fleet size
    TRIALS = 5                                # Robust sampling seed count
    PARTICLES = 30                            # Swarm / Population depth
    ITERATIONS = 70                           # Evolution / Epoch horizon
    
    # Thermal Throttling Safety: Use half available cores
    max_workers = max(1, os.cpu_count())
    print(f"❄️ Thermal Protection Active: Running on {max_workers} of {os.cpu_count()} CPU cores.")

    print(f"🚀 Initializing Unified Benchmark Suite for {LOCATION}")
    G0 = load_osm_network(LOCATION)
    print(f"✅ Base Graph Cached: {len(G0.nodes)} nodes, {len(G0.edges)} edges.\n")

    tasks = []
    for n, w, cap, veh in itertools.product(CUSTOMER_SIZES, DISTANCE_WEIGHTS, CAPACITIES, VEHICLE_COUNTS):
        for trial in range(TRIALS):
            seed = 42 + trial
            tasks.append((G0, n, seed, w, veh, cap, PARTICLES, ITERATIONS))

    total_tasks = len(tasks)
    print(f"⚡ Dispatched {total_tasks} simulation environments ({total_tasks * 3} individual runs)...")

    all_results = []
    start_bench = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_experiment_task, t) for t in tasks]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res: 
                all_results.extend(res)
            if (i + 1) % 5 == 0 or (i + 1) == total_tasks:
                print(f"   [{i + 1}/{total_tasks}] Configurations evaluated...")

    total_duration = time.perf_counter() - start_bench
    print(f"\n✅ All tasks completed successfully in {total_duration:.2f} seconds.")

    # --- Data Processing & Aggregation ---
    df = pd.DataFrame(all_results)
    if df.empty:
        raise RuntimeError("Benchmark yielded zero results. Check component errors above.")
    
    valid_df = df[df["Cost"] < 900000].copy()

    summary = valid_df.groupby(["N", "Weight", "Capacity", "Algorithm"]).agg(
        Mean_Cost=("Cost", "mean"),
        Min_Cost=("Cost", "min"),
        Mean_Dist_km=("Distance_km", "mean"),
        Mean_Time_min=("Travel_Time_min", "mean"),
        Mean_Exec_s=("Time_s", "mean")
    ).reset_index()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    print("\n🏆 Unified Performance & Speed Benchmark Summary:")
    print("========================================================================================")
    print(summary.to_string(index=False))
    print("========================================================================================")

    output_filename = "vrp_unified_benchmark_results.csv"
    summary.to_csv(output_filename, index=False)
    print(f"\n📁 Detailed summary successfully exported to: {output_filename}")

if __name__ == "__main__":
    run_benchmarks()