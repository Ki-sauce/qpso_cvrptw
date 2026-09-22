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
    rng = np.random.default_rng(seed)
    components = nx.strongly_connected_components(G) if G.is_directed() else nx.connected_components(G)
    nodes = np.asarray(list(max(components, key=len)))
    if len(nodes) < n + 1:
        raise ValueError(f"Network requires {n + 1} connected nodes.")
    chosen = rng.choice(nodes, size=n + 1, replace=False)
    return int(chosen[0]), [int(x) for x in chosen[1:]]

def run_experiment_task(task_params):
    G0, n, seed, dist_weight, vehicles, capacity, particles, iterations = task_params
    results_data = []
    
    try:
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
        
        instance = VRPInstance(depot=depot, customers=customers_obj, vehicle_capacity=capacity, num_vehicles=vehicles)

        algorithms = [
            ("A*", lambda: solve_dynamic_heuristic(G, instance, method='astar', distance_weight=dist_weight)),
            ("GA", lambda: solve_ga_baseline(G, instance, particles=particles, iterations=iterations, seed=seed, distance_weight=dist_weight)),
            ("QPSO", lambda: solve_qpso(G, instance, particles=particles, iterations=iterations, seed=seed, distance_weight=dist_weight))
        ]

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
                "Cost": res['score'],
                "Time_s": exec_time
            })
    except Exception as exc:
        print(f"Benchmark task failed (N={n}, seed={seed}): {exc}")
        
    return results_data

def run_throttled_benchmarks():
    LOCATION = "Manhattan, New York, USA"
    
    # --- THERMALLY OPTIMIZED SPACE ---
    CUSTOMER_SIZES = [10, 20, 30]             # Perfect scaling curve without blowing up CPU
    DISTANCE_WEIGHTS = [0.20, 0.80]           
    CAPACITIES = [30.0, 50.0]                 
    VEHICLE_COUNTS = [4]                      
    TRIALS = 10                               # Statistically robust sample
    PARTICLES = 30                            # Fast convergence swarm depth
    ITERATIONS = 80                           # Balanced epoch horizon
    
    # THERMAL THROTTLE: Use only half available cores to keep laptop cool and responsive
    max_workers = max(1, os.cpu_count() // 2)
    print(f"❄️ Thermal Throttling Active: Using {max_workers} of {os.cpu_count()} CPU cores.")

    print(f"🚀 Initializing Throttled Benchmark Matrix for {LOCATION}")
    G0 = load_osm_network(LOCATION)
    print(f"✅ Base Graph Loaded: {len(G0.nodes)} nodes.\n")

    tasks = []
    for n, w, cap, veh in itertools.product(CUSTOMER_SIZES, DISTANCE_WEIGHTS, CAPACITIES, VEHICLE_COUNTS):
        for trial in range(TRIALS):
            seed = 42 + trial
            tasks.append((G0, n, seed, w, veh, cap, PARTICLES, ITERATIONS))

    total_tasks = len(tasks)
    print(f"⚡ Executing {total_tasks} macro-environments ({total_tasks * 3} optimizations)...")

    all_results = []
    start_time = time.perf_counter()
    
    # Controlled concurrency pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_experiment_task, t) for t in tasks]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res: all_results.extend(res)
            if (i+1) % 10 == 0 or (i+1) == total_tasks:
                print(f"   [{i+1}/{total_tasks}] Configurations processed...")

    print(f"\n✅ Completed safely in {time.perf_counter() - start_time:.2f} seconds.")

    df = pd.DataFrame(all_results)
    if df.empty:
        raise RuntimeError("No benchmark results were produced; inspect task errors above.")
    valid_df = df[df["Cost"] < 900000].copy()

    summary = valid_df.groupby(["N", "Weight", "Capacity", "Vehicles", "Algorithm"]).agg(
        Mean_Cost=("Cost", "mean"),
        Std_Cost=("Cost", "std"),
        Min_Cost=("Cost", "min"),
        Mean_Time=("Time_s", "mean")
    ).reset_index()

    print("\n🏆 Macroscopic Throttled Benchmark Summary:")
    print(summary.to_string(index=False))
    
    summary.to_csv("vrp_throttled_results.csv", index=False)
    print("\n📁 Exported to vrp_throttled_results.csv")

if __name__ == "__main__":
    run_throttled_benchmarks()
