# run_v3_benchmarks.py
import os
import time
import glob
import math
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from pyvrp import Model, stop

from core.engine import flatten_instance_for_jit
from core.jit_eval import evaluate_swarm_jit, evaluate_detailed_jit
from core.qpso import QPSO
from core.ml_warmstart import generate_warm_keys
from core.vrp import Customer, VRPInstance


class V3BenchmarkAdapter:
    def __init__(self, use_ml_warmstart=True, particles=40, iterations=200):
        self.use_ml_warmstart = use_ml_warmstart
        self.particles = particles
        self.iterations = iterations
        self._warmup_jit()

    def _warmup_jit(self):
        """Compiles and caches both Numba kernels before any timers start."""
        dummy_particles = np.random.uniform(0, 1, (2, 5)).astype(np.float32)
        dummy_demands = np.zeros(6, dtype=np.float32)
        dummy_due = np.full(6, 1000.0, dtype=np.float32)
        dummy_matrix = np.ones((6, 6), dtype=np.float32)
        np.fill_diagonal(dummy_matrix, 0.0)

        jit_args = (
            dummy_demands, 100.0, 2,
            dummy_matrix, dummy_matrix, dummy_demands, dummy_due, dummy_demands,
            0.0, 1.0, 10000.0, 0.0
        )
        evaluate_swarm_jit(dummy_particles, *jit_args)
        evaluate_detailed_jit(dummy_particles[0], *jit_args)
        print("[System] Numba JIT kernels compiled and cached.")

    def parse_solomon_instance(self, filepath):
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        veh_line = lines[3].split()
        num_vehicles = int(veh_line[0])
        capacity = int(veh_line[1])

        cust_idx = lines.index("CUSTOMER")
        node_coords = {}
        node_data = {}
        customers = []

        for line in lines[cust_idx + 2:]:
            parts = line.split()
            if len(parts) < 7:
                continue

            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            demand = int(parts[3])
            ready, due, service = int(parts[4]), int(parts[5]), int(parts[6])

            node_coords[node_id] = (x, y)
            node_data[node_id] = {
                "x": x, "y": y,
                "demand": demand, "ready": ready, "due": due, "service": service,
            }

            if node_id != 0:
                customers.append(Customer(
                    node=node_id,
                    demand=float(demand),
                    ready_time=float(ready),
                    due_time=float(due),
                    service_time=float(service),
                ))

        instance = VRPInstance(
            depot=0,
            customers=customers,
            vehicle_capacity=float(capacity),
            num_vehicles=num_vehicles,
        )

        distances_dict = {}
        times_dict = {}
        dimension = len(node_coords)

        for i in range(dimension):
            x1, y1 = node_coords[i]
            for j in range(dimension):
                dist = math.hypot(x1 - node_coords[j][0], y1 - node_coords[j][1])
                distances_dict[(i, j)] = dist
                times_dict[(i, j)] = dist

        return instance, distances_dict, times_dict, dimension, node_coords, node_data

    def evaluate_instance(self, filepath):
        instance, distances, times, dimension, coords, data = self.parse_solomon_instance(filepath)

        # Standard academic VRPTW setup:
        # Start at depot ready time (0.0), score pure Euclidean distance (distance_weight=1.0),
        # and enforce SLA time windows via lateness_penalty.
        jit_args = flatten_instance_for_jit(
            instance, distances, times,
            time_weight=0.0,
            distance_weight=1.0,
            lateness_penalty=10000.0,
            dispatch_start_s=float(data[0]["ready"]),
            depot_due_time=float(data[0]["due"]),
        )

        # Build NetworkX graph outside the tactical timer
        G = nx.DiGraph()
        for i in range(dimension):
            G.add_node(i, **data[i])
        edges = [
            (i, j, {"_routing_weight": distances[(i, j)]})
            for i in range(dimension)
            for j in range(dimension)
            if i != j
        ]
        G.add_edges_from(edges)

        # =========================================================
        # 1. YOUR ENGINE: V3 (Runs FIRST for instant feedback)
        # =========================================================
        t_start = time.perf_counter()
        warm_keys = (
            generate_warm_keys(instance, G=G, n_particles=self.particles)
            if self.use_ml_warmstart
            else None
        )
        t_tactical = time.perf_counter() - t_start

        has_warm = warm_keys is not None and len(warm_keys) > 0
        initial_pos = warm_keys[0] if has_warm else None

        # Evaluate Gear 1 (Warmstart) initial objective before QPSO
        warmstart_obj = np.nan
        if initial_pos is not None:
            ws_obj, _, _, _, _, _, _ = evaluate_detailed_jit(
                np.asarray(initial_pos, dtype=np.float32), *jit_args
            )
            warmstart_obj = float(ws_obj)

        # Gear 2: Strategic QPSO Refinement
        qpso = QPSO(jit_args=jit_args, n_particles=self.particles, iterations=self.iterations)
        t_strategic_start = time.perf_counter()

        best_pos, v3_final_score, history = qpso.optimize(
            dimensions=len(instance.customers),
            initial_position=initial_pos,
            initial_population=warm_keys if has_warm else None,
        )
        t_strategic = time.perf_counter() - t_strategic_start

        # Extract exact physical telemetry from the best particle
        (
            v3_obj,
            v3_dist,
            v3_time,
            v3_wait,
            v3_serv,
            v3_late,
            v3_veh,
        ) = evaluate_detailed_jit(np.asarray(best_pos, dtype=np.float32), *jit_args)

        # =========================================================
        # 2. BASELINE: PyVRP (Scaled by 100 for 2-decimal precision)
        # =========================================================
        # Pragmatic MacBook Air caps: 5s (N=100), 10s (N=200-400), 15s (N=600-1000)
        time_limit_s = 5 if dimension <= 101 else (10 if dimension <= 401 else 15)
        SCALE = 100

        pyvrp_start = time.perf_counter()
        m = Model()
        locations = []

        depot_loc = m.add_location(x=int(round(coords[0][0] * SCALE)), y=int(round(coords[0][1] * SCALE)))
        locations.append(depot_loc)
        m.add_depot(
            location=depot_loc,
            tw_early=int(round(data[0]["ready"] * SCALE)),
            tw_late=int(round(data[0]["due"] * SCALE)),
        )

        m.add_vehicle_type(
            num_available=instance.num_vehicles,
            capacity=int(instance.vehicle_capacity),
        )

        for i in range(1, dimension):
            loc = m.add_location(x=int(round(coords[i][0] * SCALE)), y=int(round(coords[i][1] * SCALE)))
            locations.append(loc)
            m.add_client(
                location=loc,
                delivery=data[i]["demand"],
                tw_early=int(round(data[i]["ready"] * SCALE)),
                tw_late=int(round(data[i]["due"] * SCALE)),
                service_duration=int(round(data[i]["service"] * SCALE)),
            )

        for i in range(dimension):
            for j in range(dimension):
                dist_scaled = int(round(distances[(i, j)] * SCALE))
                m.add_edge(locations[i], locations[j], distance=dist_scaled, duration=dist_scaled)

        pyvrp_result = m.solve(stop=stop.MaxRuntime(time_limit_s), display=False)
        t_pyvrp = time.perf_counter() - pyvrp_start
        pyvrp_cost = pyvrp_result.cost() / float(SCALE)

        dataset_group = Path(filepath).parent.name

        return {
            "dataset": dataset_group,
            "instance": Path(filepath).name,
            "dimension": dimension - 1,
            "v3_tactical_s": round(t_tactical, 4),
            "v3_strategic_s": round(t_strategic, 4),
            "v3_total_s": round(t_tactical + t_strategic, 4),
            "v3_warmstart_obj": round(warmstart_obj, 2),
            "v3_objective": round(float(v3_obj), 2),
            "v3_distance": round(float(v3_dist), 2),
            "v3_lateness": round(float(v3_late), 2),
            "v3_vehicles": int(v3_veh),
            "pyvrp_total_s": round(t_pyvrp, 4),
            "pyvrp_objective": round(pyvrp_cost, 2),
            "v3_vs_pyvrp_gap_%": round(((float(v3_dist) - pyvrp_cost) / max(1.0, pyvrp_cost)) * 100.0, 2),
        }


def _instance_sort_key(filepath):
    """Sorts Solomon (N=100) first, then Homberger 200 -> 400 -> 600 -> 800 -> 1000."""
    p = Path(filepath)
    parent = p.parent.name.lower()
    if "solomon" in parent:
        size_order = 100
    else:
        for sz in (200, 400, 600, 800, 1000):
            if f"_{sz}_" in parent:
                size_order = sz
                break
        else:
            size_order = 9999
    return (size_order, p.name.lower())


if __name__ == "__main__":
    adapter = V3BenchmarkAdapter(use_ml_warmstart=True, particles=40, iterations=200)
    csv_filename = "v3_benchmark_results.csv"

    # Load already-completed instances so Ctrl+C / resume never loses progress
    completed = set()
    if os.path.exists(csv_filename):
        try:
            existing_df = pd.read_csv(csv_filename)
            if "instance" in existing_df.columns:
                completed = set(existing_df["instance"].astype(str).tolist())
                print(f"[Resume] Found {len(completed)} completed instances in {csv_filename}.")
        except Exception:
            pass

    search_paths = [
        "data/Solomon/**/*.[tT][xX][tT]",
        "data/Gehring_Homberger/**/*.[tT][xX][tT]",
    ]

    all_files = []
    for pattern in search_paths:
        all_files.extend(glob.glob(pattern, recursive=True))

    all_files = sorted(all_files, key=_instance_sort_key)

    if not all_files:
        print("[Error] No instance files found. Check directory structure.")
        exit(1)

    print(f"Found {len(all_files)} total instances (sorted N=100 -> N=1000). Starting benchmark...")

    for idx, filepath in enumerate(all_files, 1):
        fname = Path(filepath).name
        if fname in completed:
            print(f"[{idx}/{len(all_files)}] Skipping {fname} (already in CSV)")
            continue

        print(f"[{idx}/{len(all_files)}] Benchmarking {fname}...")
        try:
            res = adapter.evaluate_instance(filepath)

            print(
                f"    -> V3: {res['v3_total_s']}s (Tactical: {res['v3_tactical_s']}s) | "
                f"Dist: {res['v3_distance']} | Lateness: {res['v3_lateness']} | "
                f"PyVRP ({res['pyvrp_total_s']}s): {res['pyvrp_objective']} | "
                f"Gap: {res['v3_vs_pyvrp_gap_%']}%"
            )

            write_header = not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0
            pd.DataFrame([res]).to_csv(csv_filename, mode="a", header=write_header, index=False)

        except Exception as e:
            print(f"[Warning] Failed on {filepath}: {str(e)}")
            traceback.print_exc()

    print(f"Done. Final results saved to {csv_filename}")