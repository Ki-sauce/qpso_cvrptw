import math
import numpy as np
import torch

from .graph_model import add_objective_weights, build_all_matrices_scipy
from .vrp import (
    VRPInstance, construct_feasible_order, evaluate_routes,
    optimize_route_2opt, split_random_key_solution,
)
from .qpso import QPSO
from .baselines import run_random_search, run_exact_small
from .ga import run_ga
from .ml_warmstart import generate_warm_keys, get_model
from .profiler import profiler

def _feasible_random_key(instance):
    order = construct_feasible_order(instance)
    if order is None: return None, None
    key = np.empty(len(order), dtype=float)
    denominator = max(1, len(order) - 1)
    for rank, customer_index in enumerate(order):
        key[customer_index] = rank / denominator
    return key, order

def build_optimizer(G, instance, time_weight=1.0, distance_weight=0.0):
    if time_weight < 0 or distance_weight < 0:
        raise ValueError("Objective weights must be non-negative.")

    weighted_graph = add_objective_weights(
        G, time_weight=time_weight, distance_weight=distance_weight
    )

    costs, distances, travel_times, dist_mat, time_mat, trace_path = build_all_matrices_scipy(
        weighted_graph, instance, weight="_routing_weight"
    )
    
    customers = instance.customers

    def evaluate(position, apply_quantum_annealing=False):
        if position is None: return math.inf, {"objective": math.inf}
        pos_arr = np.asarray(position)
        if pos_arr.ndim != 1 or len(pos_arr) != len(customers):
            return math.inf, {"objective": math.inf}
        try:
            if not np.all(np.isfinite(pos_arr.astype(float, copy=False))):
                return math.inf, {"objective": math.inf}
        except (TypeError, ValueError):
            return math.inf, {"objective": math.inf}
            
        is_integer_permutation = (
            np.issubdtype(pos_arr.dtype, np.integer)
            and len(np.unique(pos_arr)) == len(pos_arr)
            and set(int(value) for value in pos_arr) == set(range(len(customers)))
        )
        if is_integer_permutation:
            order_idx = [int(value) for value in pos_arr]
        else:
            order_idx = np.argsort(pos_arr, kind="stable")
            
        try:
            order = [customers[int(index)] for index in order_idx]
        except (IndexError, TypeError, ValueError):
            return math.inf, {"objective": math.inf}
            
        routes = split_random_key_solution(order, instance)
        if apply_quantum_annealing and routes is not None:
            routes = [optimize_route_2opt(route, costs) for route in routes]
            
        score, metrics = evaluate_routes(
            routes, instance, costs, distances, {},
            time_weight=time_weight, distance_weight=distance_weight,
            travel_times=travel_times,
        )
        
        used_paths = {}
        for route in (routes or []):
            for u, v in zip(route[:-1], route[1:]):
                if (u, v) not in used_paths:
                    used_paths[(u, v)] = trace_path(u, v)
                    
        return score, {"routes": routes or [], "order": order, "paths": used_paths, **metrics}

    return evaluate, trace_path, dist_mat, time_mat, costs, distances, travel_times

def _result_payload(algorithm, score, history, result, paths):
    return {
        "algorithm": algorithm, "score": score, "history": history,
        "routes": result.get("routes", []),
        "travel_time_s": result.get("travel_time_s", float("inf")),
        "distance_m": result.get("distance_m", float("inf")),
        "service_time_s": result.get("service_time_s", 0.0),
        "waiting_time_s": result.get("waiting_time_s", 0.0),
        "total_duration_s": result.get("total_duration_s", float("inf")),
        "tw_penalty": result.get("tw_penalty", 0.0),
        "total_lateness_s": result.get("total_lateness_s", 0.0),
        "paths": paths,
    }

def solve_qpso(
    G, instance, particles=30, iterations=100, seed=42,
    time_weight=1.0, distance_weight=0.0, use_ml_warmstart=True,
    precomputed_evaluate=None, precomputed_matrices=None,
    initial_position=None
):
    profiler.start_request()
    with profiler.measure("1. Graph & Matrix Export"):
        if precomputed_evaluate and precomputed_matrices:
            evaluate, _ = precomputed_evaluate
            dist_matrix, time_matrix = precomputed_matrices
        else:
            evaluate, _, dist_matrix, time_matrix, _, _, _ = build_optimizer(
                G, instance, time_weight=time_weight, distance_weight=distance_weight
            )
        
        demands = np.array([c.demand for c in instance.customers], dtype=np.float64)
        tw_starts = np.array([c.ready_time for c in instance.customers], dtype=np.float64)
        tw_ends = np.array([c.due_time for c in instance.customers], dtype=np.float64)
        service_times = np.array([c.service_time for c in instance.customers], dtype=np.float64)

    with profiler.measure("2. ML Warm-Start & Seeding"):
        warm_keys = None
        # TRUE HYBRID PIPELINE: Inject explicit Neural A* base permutation directly
        if initial_position is not None:
            initial_key = initial_position
        elif use_ml_warmstart:
            try:
                warm_keys = generate_warm_keys(instance, G=G, n_particles=particles, warm_start_fraction=0.5)
                initial_key = warm_keys[0]
            except Exception:
                initial_key, _ = _feasible_random_key(instance)
        else:
            initial_key, _ = _feasible_random_key(instance)

    with profiler.measure("3. QPSO Optimization Loop"):
        optimizer = QPSO(
            n_particles=particles, iterations=iterations, seed=seed,
            dist_matrix=dist_matrix, time_matrix=time_matrix, demands=demands,
            tw_starts=tw_starts, tw_ends=tw_ends, service_times=service_times,
            vehicle_capacity=instance.vehicle_capacity, num_vehicles=instance.num_vehicles,
            depot_idx=0, time_weight=time_weight, distance_weight=distance_weight,
        )
        try:
            best_x, score, history = optimizer.optimize(
                len(instance.customers), initial_position=initial_key,
                initial_population=warm_keys if (use_ml_warmstart and warm_keys is not None) else None
            )
        except TypeError:
            best_x, score, history = optimizer.optimize(len(instance.customers), initial_position=initial_key)

    with profiler.measure("4. Final 2-Opt Evaluation & Payload"):
        _, result = evaluate(best_x, apply_quantum_annealing=True)
        payload = _result_payload("QPSO (Neural-Hybrid)", score, history, result, result.get("paths", {}))

    print(profiler.summary())
    return payload

def solve_ga_baseline(
    G, instance, particles=40, iterations=100, seed=42,
    time_weight=1.0, distance_weight=0.0,
    precomputed_evaluate=None, precomputed_matrices=None,
    initial_permutation=None
):
    profiler.start_request()
    with profiler.measure("1. Graph & Matrix Export"):
        if precomputed_evaluate and precomputed_matrices:
            evaluate, _ = precomputed_evaluate
            dist_matrix, time_matrix = precomputed_matrices
        else:
            evaluate, _, dist_matrix, time_matrix, _, _, _ = build_optimizer(
                G, instance, time_weight=time_weight, distance_weight=distance_weight
            )

        if initial_permutation is None:
            _, initial_permutation = _feasible_random_key(instance)
            
        demands = np.array([c.demand for c in instance.customers], dtype=np.float64)
        tw_starts = np.array([c.ready_time for c in instance.customers], dtype=np.float64)
        tw_ends = np.array([c.due_time for c in instance.customers], dtype=np.float64)
        service_times = np.array([c.service_time for c in instance.customers], dtype=np.float64)
    
    with profiler.measure("2. GA Optimization Loop"):
        best_x, score, history = run_ga(
            dimensions=len(instance.customers), population_size=particles,
            iterations=iterations, seed=seed, initial_permutation=initial_permutation,
            dist_matrix=dist_matrix, time_matrix=time_matrix, demands=demands,
            tw_starts=tw_starts, tw_ends=tw_ends, service_times=service_times,
            vehicle_capacity=instance.vehicle_capacity, num_vehicles=instance.num_vehicles,
            depot_idx=0, time_weight=time_weight, distance_weight=distance_weight,
        )

    with profiler.measure("3. Final 2-Opt Evaluation & Payload"):
        _, result = evaluate(best_x, apply_quantum_annealing=True)
        payload = _result_payload("GA (Neural-Hybrid)", score, history, result, result.get("paths", {}))

    print(profiler.summary())
    return payload

def flatten_instance_for_jit(
    instance, distances_dict, travel_times_dict,
    time_weight=1.0, distance_weight=0.0,
    lateness_penalty=10000.0, dispatch_start_s=32400.0,
    depot_due_time=86400.0
):
    """Converts OOP classes and dictionaries into the exact 12-tuple expected by Numba JIT."""
    n_nodes = len(instance.customers) + 1  # Node 0 is depot

    distance_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    time_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    demands = np.zeros(n_nodes, dtype=np.float32)
    ready_times = np.zeros(n_nodes, dtype=np.float32)
    due_times = np.zeros(n_nodes, dtype=np.float32)
    service_times = np.zeros(n_nodes, dtype=np.float32)

    ready_times[0] = float(dispatch_start_s)
    due_times[0] = float(depot_due_time)

    for i in range(n_nodes):
        for j in range(n_nodes):
            distance_matrix[i, j] = distances_dict.get((i, j), np.inf)
            time_matrix[i, j] = travel_times_dict.get((i, j), np.inf)

    for c in instance.customers:
        idx = c.node
        demands[idx] = c.demand
        ready_times[idx] = c.ready_time
        due_times[idx] = c.due_time
        service_times[idx] = c.service_time

    return (
        demands,
        float(instance.vehicle_capacity),
        int(instance.num_vehicles),
        distance_matrix,
        time_matrix,
        ready_times,
        due_times,
        service_times,
        float(time_weight),
        float(distance_weight),
        float(lateness_penalty),
        float(dispatch_start_s),
    )