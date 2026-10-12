import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.graph_model import load_osm_network
from core.traffic import apply_traffic_scenario
from core.vrp import Customer, VRPInstance
from core.engine import build_optimizer, solve_qpso, solve_ga_baseline
from core.heuristics import solve_dynamic_heuristic
from core.stop_selection import choose_spread_stops
from core.ml_warmstart import predict_edge_probabilities

def run_lightning_benchmark():
    LOCATION = "Salt Lake, Kolkata, India"
    CUSTOMER_SCALES = [20,50,100,200,300,400,500]
    VEHICLES = 100
    CAPACITY = 25.0
    PARTICLES = 100
    ITERATIONS = 1000
    SEED = 42

    print(f"🌍 Loading OSM Network for {LOCATION}...")
    G0 = load_osm_network(LOCATION)
    G = apply_traffic_scenario(G0, seed=SEED, mode='simulated')

    results = []

    for n in CUSTOMER_SCALES:
        print(f"\n🚀 Initiating Sequential Hybrid Stress Test for N={n} Customers...")
        
        depot, customer_nodes = choose_spread_stops(G, n, seed=SEED)
        rng = np.random.default_rng(SEED)
        demands = rng.integers(1, 8, size=len(customer_nodes))
        
        customers_obj = [
            Customer(
                node=node, demand=float(d), 
                ready_time=32400.0, due_time=86400.0, service_time=300.0
            ) for node, d in zip(customer_nodes, demands)
        ]
        
        instance = VRPInstance(depot=depot, customers=customers_obj, vehicle_capacity=CAPACITY, num_vehicles=VEHICLES)

        t0 = time.perf_counter()
        evaluate_shared, trace_path_shared, dist_mat, time_mat, costs, dists, tt = build_optimizer(
            G, instance, time_weight=1.0, distance_weight=0.2
        )
        matrix_time = time.perf_counter() - t0
        print(f"   ✓ SciPy Matrices & Lazy Tracing Built: {matrix_time:.4f}s")

        precomputed_eval = (evaluate_shared, trace_path_shared)
        precomputed_mats = (dist_mat, time_mat)
        precomputed_dicts = (costs, dists, tt)

        # 1. Neural Prediction
        try:
            edge_probs = predict_edge_probabilities(instance, G)
        except Exception:
            edge_probs = None

        # 2. Stage 1: Neural A* Constructive Heuristic
        t_astar = time.perf_counter()
        astar_res = solve_dynamic_heuristic(
            G, instance, method='astar', distance_weight=0.2, 
            trace_path_func=trace_path_shared, precomputed_dicts=precomputed_dicts,
            edge_probs=edge_probs
        )
        astar_time = time.perf_counter() - t_astar
        print(f"   ✓ Neural A* Finished -> Score: {astar_res['score']:<10.2f} | Time: {astar_time:.3f}s")
        results.append({"N": n, "Algorithm": "A*", "Score": astar_res['score'], "Matrix_Time_s": matrix_time, "Solver_Time_s": astar_time})

        # 3. Stage 2: Metaheuristic Handoff Extraction
        node_to_idx = {c.node: i for i, c in enumerate(instance.customers)}
        base_permutation = []
        for route in astar_res["routes"]:
            for node in route:
                if node != instance.depot:
                    base_permutation.append(node_to_idx[node])
                    
        qpso_base_pos = np.zeros(len(instance.customers), dtype=float)
        denominator = max(1, len(instance.customers) - 1)
        for rank, cust_idx in enumerate(base_permutation):
            qpso_base_pos[cust_idx] = rank / denominator

        # 4. Stage 3: Metaheuristic Local Search Execution
        def run_ga():
            t = time.perf_counter()
            res = solve_ga_baseline(
                G, instance, particles=PARTICLES, iterations=ITERATIONS, seed=SEED, 
                distance_weight=0.2, precomputed_evaluate=precomputed_eval, precomputed_matrices=precomputed_mats, 
                initial_permutation=base_permutation
            )
            return "GA", res, time.perf_counter() - t

        def run_qpso():
            t = time.perf_counter()
            res = solve_qpso(
                G, instance, particles=PARTICLES, iterations=ITERATIONS, seed=SEED, 
                distance_weight=0.2, use_ml_warmstart=False, precomputed_evaluate=precomputed_eval, precomputed_matrices=precomputed_mats, 
                initial_position=qpso_base_pos
            )
            return "QPSO", res, time.perf_counter() - t

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_ga), executor.submit(run_qpso)]
            for future in as_completed(futures):
                algo, res, exec_time = future.result()
                print(f"   ✓ {algo:<5} Finished -> Score: {res['score']:<10.2f} | Time: {exec_time:.3f}s")
                results.append({
                    "N": n, "Algorithm": algo, "Score": res['score'], 
                    "Matrix_Time_s": matrix_time, "Solver_Time_s": exec_time
                })

    df = pd.DataFrame(results)
    df.to_csv("vrp_v3_quantum_benchmark1.csv", index=False)
    print("\n💾 Benchmark saved to vrp_v3_quantum_benchmark.csv")

if __name__ == "__main__":
    run_lightning_benchmark()