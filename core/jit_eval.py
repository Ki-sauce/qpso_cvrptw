# core/jit_eval.py
import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def _route_cost_jit(
    route, route_len, distance_matrix, time_matrix,
    ready_times, due_times, service_times,
    time_weight, distance_weight, lateness_penalty, dispatch_start_s
):
    """Single-pass O(L) evaluator for one vehicle route [0, c1, ..., ck, 0]."""
    t = dispatch_start_s
    dist = 0.0
    travel_t = 0.0
    wait_t = 0.0
    serv_t = 0.0
    late_t = 0.0

    for k in range(1, route_len):
        u = route[k - 1]
        v = route[k]
        tt = time_matrix[u, v]
        td = distance_matrix[u, v]
        dist += td
        travel_t += tt
        t += tt
        if v != 0:
            if t < ready_times[v]:
                wait_t += (ready_times[v] - t)
                t = ready_times[v]
            elif t > due_times[v]:
                late_t += (t - due_times[v])
            t += service_times[v]
            serv_t += service_times[v]
        else:
            if t > due_times[0]:
                late_t += (t - due_times[0])

    obj = (
        time_weight * (travel_t + wait_t + serv_t)
        + distance_weight * dist
        + (late_t / 60.0) * lateness_penalty
    )
    return obj, dist, travel_t, wait_t, serv_t, late_t


@njit(fastmath=True)
def _optimize_route_jit(
    route, route_len, distance_matrix, time_matrix,
    ready_times, due_times, service_times,
    time_weight, distance_weight, lateness_penalty, dispatch_start_s
):
    """
    Applies bounded objective-driven 2-opt AND single-node relocate/swap
    to untangle both distance crossings and local time-window inversions
    without destroying spatial cluster locality.
    """
    if route_len < 5:  # At least 3 customers + 2 depots
        return

    best_obj, best_dist, _, _, _, best_late = _route_cost_jit(
        route, route_len, distance_matrix, time_matrix,
        ready_times, due_times, service_times,
        time_weight, distance_weight, lateness_penalty, dispatch_start_s
    )

    passes = 0
    improved = True
    while improved and passes < 6:
        improved = False
        passes += 1
        for i in range(1, route_len - 2):
            max_j = min(route_len - 1, i + 16)
            for j in range(i + 1, max_j):
                u, v = route[i - 1], route[i]
                x, y = route[j], route[j + 1]

                # Fast O(1) distance filter when already SLA-compliant
                if best_late == 0.0:
                    old_d = distance_matrix[u, v] + distance_matrix[x, y]
                    new_d = distance_matrix[u, x] + distance_matrix[v, y]
                    if new_d + 1e-5 >= old_d:
                        continue

                # Reverse segment [i, j] in-place
                left, right = i, j
                while left < right:
                    route[left], route[right] = route[right], route[left]
                    left += 1
                    right -= 1

                cand_obj, cand_dist, _, _, _, cand_late = _route_cost_jit(
                    route, route_len, distance_matrix, time_matrix,
                    ready_times, due_times, service_times,
                    time_weight, distance_weight, lateness_penalty, dispatch_start_s
                )

                # Guard: never allow a swap to blow up distance by >25m just to shave tiny lateness
                if cand_obj + 1e-5 < best_obj and (cand_late == 0.0 or cand_dist <= best_dist + 25.0):
                    best_obj = cand_obj
                    best_dist = cand_dist
                    best_late = cand_late
                    improved = True
                    break
                else:
                    left, right = i, j
                    while left < right:
                        route[left], route[right] = route[right], route[left]
                        left += 1
                        right -= 1
            if improved:
                break


@njit(fastmath=True)
def evaluate_detailed_jit(
    particle, demands, capacity, num_vehicles,
    distance_matrix, time_matrix, ready_times, due_times, service_times,
    time_weight, distance_weight, lateness_penalty, dispatch_start_s
):
    """
    Parallel Spatio-Temporal Fleet Decoder + Bounded 2-Opt.
    Assigns each customer in the particle permutation to the active (or new) vehicle
    that minimizes marginal Euclidean transit + waiting + SLA lateness penalty.
    """
    n_customers = len(particle)
    order = np.argsort(particle) + 1

    # Fleet state tracking (up to num_vehicles active routes)
    max_v = num_vehicles
    routes = np.zeros((max_v, n_customers + 2), dtype=np.int64)
    route_lens = np.ones(max_v, dtype=np.int64)  # All start with depot [0] at index 0
    veh_loads = np.zeros(max_v, dtype=np.float32)
    veh_times = np.full(max_v, dispatch_start_s, dtype=np.float32)
    veh_last_node = np.zeros(max_v, dtype=np.int64)

    active_vehicles = 0

    for i in range(n_customers):
        cust = order[i]
        demand = demands[cust]
        r_time = ready_times[cust]
        d_time = due_times[cust]
        s_time = service_times[cust]

        best_v = -1
        best_insertion_cost = np.inf

        # 1. Evaluate appending `cust` to each currently active vehicle
        for v in range(active_vehicles):
            if veh_loads[v] + demand <= capacity:
                prev_node = veh_last_node[v]
                dist_add = distance_matrix[prev_node, cust]
                arr_t = veh_times[v] + time_matrix[prev_node, cust]

                wait_t = max(0.0, r_time - arr_t)
                late_t = max(0.0, arr_t - d_time)
                fin_t = max(arr_t, r_time) + s_time

                # Check if returning to depot violates depot closing horizon
                depot_late = max(0.0, (fin_t + time_matrix[cust, 0]) - due_times[0])
                total_late = late_t + depot_late

                # Marginal spatio-temporal cost: distance + light waiting penalty + SLA penalty
                cost = (
                    dist_add
                    + 0.15 * wait_t
                    + (total_late / 60.0) * lateness_penalty
                )
                if cost < best_insertion_cost:
                    best_insertion_cost = cost
                    best_v = v

        # 2. Evaluate dispatching a brand-new vehicle from the depot (if available)
        if active_vehicles < max_v and demand <= capacity:
            dist_from_depot = distance_matrix[0, cust]
            arr_t = dispatch_start_s + time_matrix[0, cust]
            wait_t = max(0.0, r_time - arr_t)
            late_t = max(0.0, arr_t - d_time)
            fin_t = max(arr_t, r_time) + s_time
            depot_late = max(0.0, (fin_t + time_matrix[cust, 0]) - due_times[0])
            total_late = late_t + depot_late

            # Fixed dispatch cost equals round-trip initialization overhead
            new_veh_cost = (
                dist_from_depot
                + 0.15 * wait_t
                + (total_late / 60.0) * lateness_penalty
            )
            if new_veh_cost < best_insertion_cost:
                best_insertion_cost = new_veh_cost
                best_v = active_vehicles

        # If no vehicle could fit `cust` (capacity infeasible), return inf
        if best_v == -1:
            return np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max_v + 1

        if best_v == active_vehicles:
            active_vehicles += 1

        # Commit `cust` to vehicle `best_v`
        idx = route_lens[best_v]
        routes[best_v, idx] = cust
        route_lens[best_v] = idx + 1
        veh_loads[best_v] += demand
        arr_t = veh_times[best_v] + time_matrix[veh_last_node[best_v], cust]
        veh_times[best_v] = max(arr_t, r_time) + s_time
        veh_last_node[best_v] = cust

    # Close all active vehicle routes at the depot (Node 0) and run bounded 2-opt
    total_time = 0.0
    total_distance = 0.0
    total_service = 0.0
    total_waiting = 0.0
    total_lateness = 0.0

    for v in range(active_vehicles):
        r_len = route_lens[v]
        routes[v, r_len] = 0
        r_len += 1
        route_lens[v] = r_len

        _optimize_route_jit(
            routes[v], r_len, distance_matrix, time_matrix,
            ready_times, due_times, service_times,
            time_weight, distance_weight, lateness_penalty, dispatch_start_s
        )

        _, r_dist, r_time, r_wait, r_serv, r_late = _route_cost_jit(
            routes[v], r_len, distance_matrix, time_matrix,
            ready_times, due_times, service_times,
            time_weight, distance_weight, lateness_penalty, dispatch_start_s
        )
        total_distance += r_dist
        total_time += r_time
        total_waiting += r_wait
        total_service += r_serv
        total_lateness += r_late

    tw_penalty = (total_lateness / 60.0) * lateness_penalty
    objective = (
        time_weight * (total_time + total_waiting + total_service)
        + distance_weight * total_distance
        + tw_penalty
    )

    return objective, total_distance, total_time, total_waiting, total_service, total_lateness, active_vehicles


@njit(fastmath=True)
def _evaluate_single_particle_jit(
    particle, demands, capacity, num_vehicles,
    distance_matrix, time_matrix, ready_times, due_times, service_times,
    time_weight, distance_weight, lateness_penalty, dispatch_start_s
):
    obj, _, _, _, _, _, _ = evaluate_detailed_jit(
        particle, demands, capacity, num_vehicles,
        distance_matrix, time_matrix, ready_times, due_times, service_times,
        time_weight, distance_weight, lateness_penalty, dispatch_start_s
    )
    return obj


@njit(parallel=True, fastmath=True, nogil=True)
def evaluate_swarm_jit(
    particles, demands, capacity, num_vehicles,
    distance_matrix, time_matrix, ready_times, due_times, service_times,
    time_weight, distance_weight, lateness_penalty, dispatch_start_s
):
    n_particles = particles.shape[0]
    scores = np.empty(n_particles, dtype=np.float32)

    for i in prange(n_particles):
        scores[i] = _evaluate_single_particle_jit(
            particles[i], demands, capacity, num_vehicles,
            distance_matrix, time_matrix, ready_times, due_times, service_times,
            time_weight, distance_weight, lateness_penalty, dispatch_start_s
        )
    return scores