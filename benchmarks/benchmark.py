import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.graph_model import load_osm_network
from core.traffic import apply_traffic_scenario
from core.vrp import Customer, VRPInstance
from core.engine import build_optimizer, solve_qpso, solve_ga_baseline
from core.heuristics import solve_dynamic_heuristic
from core.stop_selection import choose_spread_stops
from core.device import get_device


# =====================================================================
# OLD ML MODEL & FEATURE EXTRACTOR (1D Random Key Predictor)
# =====================================================================
class TransformerKeyPredictor(nn.Module):
    def __init__(self, feat_dim=8, d_model=128, n_heads=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.transformer(h)
        return self.head(h).squeeze(-1)


def extract_features_old(instance, G):
    depot_node = instance.depot
    depot_coords = np.array([G.nodes[depot_node]['x'], G.nodes[depot_node]['y']], dtype=np.float32)
    cust_coords = np.array([[G.nodes[c.node]['x'], G.nodes[c.node]['y']] for c in instance.customers], dtype=np.float32)
    tw_ends = np.array([c.due_time for c in instance.customers], dtype=np.float32)
    
    rel_x = (cust_coords[:, 0] - depot_coords[0]) / 100.0
    rel_y = (cust_coords[:, 1] - depot_coords[1]) / 100.0
    depot_dist = np.linalg.norm(cust_coords - depot_coords, axis=-1) / 100.0
    max_horizon = tw_ends.max() if len(tw_ends) > 0 else 86400.0

    return np.stack([
        cust_coords[:, 0] / 100.0, cust_coords[:, 1] / 100.0,
        rel_x, rel_y, np.array([c.demand for c in instance.customers], dtype=np.float32) / instance.vehicle_capacity,
        np.array([c.ready_time for c in instance.customers], dtype=np.float32) / max_horizon,
        tw_ends / max_horizon, depot_dist
    ], axis=-1).astype(np.float32)


def generate_old_warm_keys(instance, G, n_particles=100, warm_start_fraction=0.5, seed=42):
    device = get_device()
    torch.manual_seed(seed)
    model = TransformerKeyPredictor().to(device)
    model.eval()
    
    features = extract_features_old(instance, G)
    x_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        base_key = model(x_tensor).squeeze(0).cpu().numpy()

    n_dims = len(base_key)
    scale_factor = min(1.0, 30.0 / max(10, n_dims))
    effective_fraction = warm_start_fraction * scale_factor
    n_warm = int(n_particles * effective_fraction)
    
    noise_sigma = max(0.01, 0.15 / np.sqrt(max(10, n_dims)))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_sigma, size=(n_warm, n_dims))
    
    warm_keys = np.clip(base_key[None, :] + noise, 0.0, 1.0).astype(np.float32)
    return warm_keys, base_key


# =====================================================================
# MULTI-SEED OLD BENCHMARK HARNESS
# =====================================================================
def run_old_benchmark():
    LOCATION = "Salt Lake, Kolkata, India"
    CUSTOMER_SCALES = [20, 50, 100, 200, 300, 400, 500]
    SEEDS = [42, 101, 2024]
    VEHICLES = 100
    CAPACITY = 25.0
    PARTICLES = 200
    ITERATIONS = 200

    print(f"🌍 [OLD API] Loading OSM Network for {LOCATION}...")
    G0 = load_osm_network(LOCATION)

    results = []

    for seed in SEEDS:
        print(f"\n=====================================================================")
        print(f"🌱 RUNNING MULTI-SEED ITERATION WITH SEED: {seed}")
        print(f"=====================================================================")
        
        G = apply_traffic_scenario(G0, seed=seed, mode='simulated')

        for n in CUSTOMER_SCALES:
            print(f"\n🚀 [OLD API | Seed {seed}] Initiating Stress Test for N={n} Customers...")
            
            depot, customer_nodes = choose_spread_stops(G, n, seed=seed)
            rng = np.random.default_rng(seed)
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

            precomputed_eval = (evaluate_shared, trace_path_shared)
            precomputed_mats = (dist_mat, time_mat)
            precomputed_dicts = (costs, dists, tt)

            # 1. Pure Dijkstra Heuristic
            t_dijk = time.perf_counter()
            dijk_res = solve_dynamic_heuristic(
                G, instance, method='dijkstra', distance_weight=0.2, 
                trace_path_func=trace_path_shared, precomputed_dicts=precomputed_dicts
            )
            dijk_time = time.perf_counter() - t_dijk
            print(f"   ✓ Standard Dijkstra Finished -> Score: {dijk_res['score']:<10.2f} | Time: {dijk_time:.3f}s")
            results.append({
                "Seed": seed, "N": n, "Algorithm": "Dijkstra", 
                "Score": dijk_res['score'], 
                "Matrix_Time_s": matrix_time, "Solver_Time_s": dijk_time
            })

            # 2. Extract Old Warm-Start Keys
            _, base_key = generate_old_warm_keys(instance, G, n_particles=PARTICLES, seed=seed)
            base_perm = list(np.argsort(base_key))

            # 3. Parallel Metaheuristics Execution
            def run_ga():
                t = time.perf_counter()
                try:
                    res = solve_ga_baseline(
                        G, instance, particles=PARTICLES, iterations=ITERATIONS, seed=seed, 
                        distance_weight=0.2, precomputed_evaluate=precomputed_eval, precomputed_matrices=precomputed_mats, 
                        initial_permutation=base_perm
                    )
                except TypeError:
                    res = solve_ga_baseline(
                        G, instance, particles=PARTICLES, iterations=ITERATIONS, seed=seed, 
                        distance_weight=0.2, precomputed_evaluate=precomputed_eval, precomputed_matrices=precomputed_mats
                    )
                return "GA", res, time.perf_counter() - t

            def run_qpso():
                t = time.perf_counter()
                try:
                    res = solve_qpso(
                        G, instance, particles=PARTICLES, iterations=ITERATIONS, seed=seed, 
                        distance_weight=0.2, use_ml_warmstart=False, precomputed_evaluate=precomputed_eval, precomputed_matrices=precomputed_mats, 
                        initial_position=base_key
                    )
                except TypeError:
                    res = solve_qpso(
                        G, instance, particles=PARTICLES, iterations=ITERATIONS, seed=seed, 
                        distance_weight=0.2, use_ml_warmstart=True, precomputed_evaluate=precomputed_eval, precomputed_matrices=precomputed_mats
                    )
                return "QPSO", res, time.perf_counter() - t

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(run_ga), executor.submit(run_qpso)]
                for future in as_completed(futures):
                    algo, res, exec_time = future.result()
                    print(f"   ✓ {algo:<5} Finished -> Score: {res['score']:<10.2f} | Time: {exec_time:.3f}s")
                    results.append({
                        "Seed": seed, "N": n, "Algorithm": algo, 
                        "Score": res['score'], 
                        "Matrix_Time_s": matrix_time, "Solver_Time_s": exec_time
                    })

    df = pd.DataFrame(results)
    df.to_csv("vrp_old_multiseed_benchmark.csv", index=False)
    print("\n💾 Benchmark saved to vrp_old_multiseed_benchmark.csv")

    # Print summary statistics across all seeds
    print("\n=====================================================================")
    print("📊 SUMMARY STATISTICS ACROSS ALL SEEDS (OLD API)")
    print("=====================================================================")
    summary = df.groupby(["N", "Algorithm"]).agg(
        Mean_Score=("Score", "mean"),
        Std_Score=("Score", "std"),
        Mean_Solver_Time=("Solver_Time_s", "mean")
    ).reset_index()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_old_benchmark()