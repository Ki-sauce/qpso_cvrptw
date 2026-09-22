import math
import numpy as np
import torch

from .graph_model import add_objective_weights, build_stop_matrix, path_metrics
from .vrp import (
    VRPInstance,
    construct_feasible_order,
    evaluate_routes,
    optimize_route_2opt,
    split_random_key_solution,
)
from .qpso import QPSO
from .baselines import run_random_search, run_exact_small
from .ga import run_ga
from .ml_warmstart import generate_warm_keys, get_model


def _feasible_random_key(instance):
    order = construct_feasible_order(instance)
    if order is None:
        return None, None
    key = np.empty(len(order), dtype=float)
    denominator = max(1, len(order) - 1)
    for rank, customer_index in enumerate(order):
        key[customer_index] = rank / denominator
    return key, order


def build_optimizer(G, instance, time_weight=1.0, distance_weight=0.0):
    """Build one route evaluator shared by all algorithms."""
    if time_weight < 0 or distance_weight < 0:
        raise ValueError("Objective weights must be non-negative.")

    stop_nodes = [instance.depot] + [c.node for c in instance.customers]
    weighted_graph = add_objective_weights(
        G, time_weight=time_weight, distance_weight=distance_weight
    )
    costs, paths = build_stop_matrix(
        weighted_graph, stop_nodes, weight="_routing_weight"
    )

    distances = {}
    travel_times = {}
    for pair, path in paths.items():
        travel_time, distance = path_metrics(
            weighted_graph, path, weight="_routing_weight"
        )
        travel_times[pair] = travel_time
        distances[pair] = distance

    customers = instance.customers

    def evaluate(position, apply_quantum_annealing=False):
        if position is None:
            return math.inf, {"objective": math.inf}

        pos_arr = np.asarray(position)
        if pos_arr.ndim != 1 or len(pos_arr) != len(customers):
            return math.inf, {"objective": math.inf}
        try:
            if not np.all(np.isfinite(pos_arr.astype(float, copy=False))):
                return math.inf, {"objective": math.inf}
        except (TypeError, ValueError):
            return math.inf, {"objective": math.inf}

        # The GA and exact solver use integer permutations. QPSO/random search
        # use continuous random keys whose sorted order is the permutation.
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
            routes,
            instance,
            costs,
            distances,
            paths,
            time_weight=time_weight,
            distance_weight=distance_weight,
            travel_times=travel_times,
        )
        return score, {
            "routes": routes or [],
            "order": order,
            **metrics,
        }

    return evaluate, paths


def _result_payload(algorithm, score, history, result, paths):
    return {
        "algorithm": algorithm,
        "score": score,
        "history": history,
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
):
    """
    QPSO Solver integrated with the trained Transformer warm-start model.
    """
    evaluate, paths = build_optimizer(
        G, instance, time_weight=time_weight, distance_weight=distance_weight
    )
    
    initial_key = None
    warm_keys = None

    # Try applying ML warm-start if enabled
    if use_ml_warmstart:
        try:
            warm_keys = generate_warm_keys(
                instance, 
                G=G,  
                n_particles=particles, 
                warm_start_fraction=0.5
            )
            initial_key = warm_keys[0]
        except Exception as e:
            print(f"Warning: ML warm-start failed ({e}). Falling back to heuristic key.")
            initial_key, _ = _feasible_random_key(instance)
    else:
        initial_key, _ = _feasible_random_key(instance)

    qpso_eval = lambda pos: evaluate(pos, apply_quantum_annealing=True)
    
    optimizer = QPSO(
        evaluate=qpso_eval,
        n_particles=particles,
        iterations=iterations,
        seed=seed,
    )

    # Pass the full warm-started population array into QPSO for multi-particle priming
    try:
        best_x, score, history = optimizer.optimize(
            len(instance.customers), 
            initial_position=initial_key,
            initial_population=warm_keys if use_ml_warmstart else None
        )
    except TypeError:
        best_x, score, history = optimizer.optimize(
            len(instance.customers), 
            initial_position=initial_key
        )

    _, result = qpso_eval(best_x)
    return _result_payload("QPSO (ML-Warmstarted)", score, history, result, paths)


def solve_ga_baseline(
    G, instance, particles=40, iterations=100, seed=42,
    time_weight=1.0, distance_weight=0.0,
):
    evaluate, paths = build_optimizer(
        G, instance, time_weight=time_weight, distance_weight=distance_weight
    )
    _, initial_permutation = _feasible_random_key(instance)
    
    # FIX APPLIED: Enable 2-opt local search for GA just like QPSO
    ga_eval = lambda pos: evaluate(pos, apply_quantum_annealing=True)
    
    best_x, score, history = run_ga(
        evaluate_fn=ga_eval,
        dimensions=len(instance.customers),
        population_size=particles,
        iterations=iterations,
        seed=seed,
        initial_permutation=initial_permutation,
    )
    _, result = ga_eval(best_x)
    return _result_payload("Genetic Algorithm", score, history, result, paths)


def solve_random(
    G, instance, iterations=1000, seed=42,
    time_weight=1.0, distance_weight=0.0,
):
    evaluate, paths = build_optimizer(
        G, instance, time_weight=time_weight, distance_weight=distance_weight
    )
    initial_key, _ = _feasible_random_key(instance)
    best_x, score, history = run_random_search(
        evaluate, len(instance.customers), iterations, seed, initial_x=initial_key
    )
    _, result = evaluate(best_x)
    return _result_payload("Random Search", score, history, result, paths)


def solve_exact(
    G, instance, max_customers=9,
    time_weight=1.0, distance_weight=0.0,
):
    evaluate, paths = build_optimizer(
        G, instance, time_weight=time_weight, distance_weight=distance_weight
    )
    best_x, score = run_exact_small(
        evaluate, instance.customers, max_customers=max_customers
    )
    _, result = evaluate(best_x)
    return _result_payload("Exact Enumeration", score, [score], result, paths)