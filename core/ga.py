import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from core.vrp import evaluate_routes_kernel, optimize_route_2opt

def _ga_thread_evaluate(args):
    """Fast thread evaluation for GA using Numba native kernel."""
    (
        individual_permutation, dist_matrix, time_matrix, 
        demands, tw_starts, tw_ends, service_times, 
        v_cap, n_veh, d_idx, 
        t_wt, d_wt, disp_s, lat_pen
    ) = args

    route_lens_list = []
    route_flat_list = [int(d_idx)]
    curr_load = 0.0
    current_route_len = 1

    for cust_idx in individual_permutation:
        c_idx = int(cust_idx)
        c_demand = float(demands[c_idx])
        matrix_node_idx = c_idx + 1

        if curr_load + c_demand <= float(v_cap):
            route_flat_list.append(matrix_node_idx)
            curr_load += c_demand
            current_route_len += 1
        else:
            route_flat_list.append(int(d_idx))
            current_route_len += 1
            route_lens_list.append(current_route_len)
            
            # Start new route with depot and first customer (exactly 2 nodes)
            route_flat_list.append(int(d_idx))
            route_flat_list.append(matrix_node_idx)
            curr_load = c_demand
            current_route_len = 2  # ⚡ FIXED: Exactly 2 nodes!

    route_flat_list.append(int(d_idx))
    current_route_len += 1
    route_lens_list.append(current_route_len)

    while len(route_lens_list) < int(n_veh):
        route_flat_list.extend([int(d_idx), int(d_idx)])
        route_lens_list.append(2)

    route_flat_arr = np.array(route_flat_list, dtype=np.int64)
    route_lens_arr = np.array(route_lens_list, dtype=np.int64)

    obj, t_time, d_m, s_time, w_time, tw_p, lat = evaluate_routes_kernel(
        route_flat_arr, route_lens_arr, demands, tw_starts, tw_ends, service_times,
        dist_matrix, time_matrix, float(v_cap), float(t_wt), float(d_wt),
        float(disp_s), float(lat_pen)
    )
    return float(obj), route_flat_list

def _order_crossover(parent1, parent2, rng):
    n = len(parent1)
    if n < 2: return parent1.copy(), parent2.copy()
    left, right = sorted(rng.choice(n, 2, replace=False))
    def make_child(first, second):
        child = np.full(n, -1, dtype=int)
        child[left:right + 1] = first[left:right + 1]
        used = set(child[left:right + 1])
        insert_at = (right + 1) % n
        for value in np.concatenate((second[right + 1:], second[:right + 1])):
            if value not in used:
                child[insert_at] = value
                used.add(int(value))
                insert_at = (insert_at + 1) % n
        return child
    return make_child(parent1, parent2), make_child(parent2, parent1)

def _mutate_swap(permutation, rng):
    if len(permutation) > 1:
        left, right = rng.choice(len(permutation), 2, replace=False)
        permutation[left], permutation[right] = permutation[right], permutation[left]

def run_ga(
    dimensions=0, population_size=40, iterations=100, mutation_rate=0.1, seed=42,
    initial_permutation=None, max_workers=None, dist_matrix=None, time_matrix=None,
    demands=None, tw_starts=None, tw_ends=None, service_times=None, vehicle_capacity=None,
    num_vehicles=None, depot_idx=0, time_weight=1.0, distance_weight=0.0
):
    if dimensions < 1: raise ValueError("GA requires at least one dimension.")
    max_workers = max_workers or min(os.cpu_count() or 2, 8)
    rng = np.random.default_rng(seed)

    population = np.array([rng.permutation(dimensions) for _ in range(population_size)], dtype=int)
    if initial_permutation is not None:
        population[0] = np.asarray(initial_permutation, dtype=int)

    global_best_cost = np.inf
    global_best_position = None
    history = []

    worker_args_template = (
        dist_matrix, time_matrix, demands, tw_starts, tw_ends, service_times,
        float(vehicle_capacity), int(num_vehicles), int(depot_idx),
        float(time_weight), float(distance_weight), 32400.0, 10000.0
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(iterations):
            worker_args = [(ind,) + worker_args_template for ind in population]
            fitness_results = list(executor.map(_ga_thread_evaluate, worker_args))
            fitness_scores = np.array([res[0] for res in fitness_results])

            best_idx = int(np.argmin(fitness_scores))
            if global_best_position is None or fitness_scores[best_idx] < global_best_cost:
                global_best_cost = float(fitness_scores[best_idx])
                global_best_position = population[best_idx].copy()
            history.append(global_best_cost)

            def tournament():
                competitors = rng.choice(population_size, 3, replace=False)
                return population[competitors[np.argmin(fitness_scores[competitors])]]

            new_population = [global_best_position.copy()]
            while len(new_population) < population_size:
                p1, p2 = tournament().copy(), tournament().copy()
                c1, c2 = _order_crossover(p1, p2, rng)
                if rng.random() < mutation_rate: _mutate_swap(c1, rng)
                if rng.random() < mutation_rate: _mutate_swap(c2, rng)
                new_population.extend((c1, c2))
            population = np.array(new_population[:population_size], dtype=int)

    return global_best_position, global_best_cost, history