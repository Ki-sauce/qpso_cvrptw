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
        
        instance = VRPInstance(
            depot=depot, 
            customers=customers_obj, 
            vehicle_capacity=capacity, 
            num_vehicles=vehicles
        )

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
                "Cost": res.get('score', float('inf')),
                "Distance_km": res.get('distance_m', float('inf')) / 1000.0,
                "Travel_Time_min": res.get('travel_time_s', float('inf')) / 60.0,
                "Time_s": exec_time
            })
            
    except Exception as exc:
        print(f"⚠️ Max Power Task failed (N={n}, seed={seed}): {exc}")
        
    return results_data

def run_max_power_benchmarks():
    LOCATION = "Manhattan, New York, USA"
    
    # --- MAX POWER STRESS TEST HYPERPARAMETERS ---
    CUSTOMER_SIZES = [70, 80, 90]                 # Heavy enterprise load scaling
    DISTANCE_WEIGHTS = [0.50]                 # Balanced baseline weight
    CAPACITIES = [60.0]                       # Scaled payload capacity
    VEHICLE_COUNTS = [5]                      # Expanded fleet size
    TRIALS = 5                                # Statistical samples
    PARTICLES = 40                            # Heavy swarm scale
    ITERATIONS = 100                          # Full epoch depth
    
    # UNTHROTTLED: Maximum core allocation
    max_workers = os.cpu_count()
    print(f"🔥 UNTHROTTLED MAX POWER ACTIVE: Firing up all {max_workers} Apple M1 cores.")

    G0 = load_osm_network(LOCATION)
    print(f"✅ Graph Loaded: {len(G0.nodes)} nodes. Dispatching heavy compute tasks...\n")

    tasks = []
    for n, w, cap, veh in itertools.product(CUSTOMER_SIZES, DISTANCE_WEIGHTS, CAPACITIES, VEHICLE_COUNTS):
        for trial in range(TRIALS):
            seed = 42 + trial
            tasks.append((G0, n, seed, w, veh, cap, PARTICLES, ITERATIONS))

    total_tasks = len(tasks)
    print(f"⚡ Running {total_tasks} heavy environments ({total_tasks * 3} total high-depth optimizations)...")

    all_results = []
    start_bench = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_experiment_task, t) for t in tasks]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res: 
                all_results.extend(res)
            print(f"   [{i + 1}/{total_tasks}] Heavy configuration completed.")

    total_duration = time.perf_counter() - start_bench
    print(f"\n✅ Max Power workload finished in {total_duration:.2f} seconds.")

    df = pd.DataFrame(all_results)
    if df.empty:
        raise RuntimeError("Benchmark yielded zero results.")
    
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

    print("\n🏆 Max Power Stress Test Summary ($N=30$ and $N=40$):")
    print("========================================================================================")
    print(summary.to_string(index=False))
    print("========================================================================================")

    output_filename = "vrp_max_power_results_max.csv"
    summary.to_csv(output_filename, index=False)
    print(f"\n📁 Exported to: {output_filename}")

if __name__ == "__main__":
    run_max_power_benchmarks()