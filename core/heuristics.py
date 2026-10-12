import math
import numpy as np
import networkx as nx
from .graph_model import add_objective_weights, path_metrics
from .vrp import (
    construct_feasible_order, evaluate_routes, split_random_key_solution,
)

def _straight_line_distance_m(G, n1, n2):
    x1, y1 = G.nodes[n1].get("x"), G.nodes[n1].get("y")
    x2, y2 = G.nodes[n2].get("x"), G.nodes[n2].get("y")
    if None in (x1, y1, x2, y2): return 0.0
    crs = G.graph.get("crs")
    is_geographic = getattr(crs, "is_geographic", False)
    if not is_geographic:
        is_geographic = (abs(float(x1)) <= 180 and abs(float(x2)) <= 180 and abs(float(y1)) <= 90 and abs(float(y2)) <= 90)
    if is_geographic:
        radius = 6_371_000.0
        lat1, lat2 = math.radians(float(y1)), math.radians(float(y2))
        dlat, dlon = lat2 - lat1, math.radians(float(x2) - float(x1))
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * radius * math.asin(math.sqrt(min(1.0, a)))
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))

def _max_edge_speed_mps(G):
    speeds = []
    for u, v, data in G.edges(data=True):
        distance = float(data.get("distance_m", data.get("length", 0.0)))
        time_s = float(data.get("travel_time_s", 0.0))
        if distance > 0 and time_s > 0 and math.isfinite(distance / time_s):
            speeds.append(distance / time_s)
    return max(speeds, default=0.0)

def solve_dynamic_heuristic(
    G, instance, method="dijkstra", time_weight=1.0, distance_weight=0.0,
    trace_path_func=None, precomputed_dicts=None, edge_probs=None, neural_beta=1200.0
):
    demands = {c.node: c.demand for c in instance.customers}
    customers = {c.node: c for c in instance.customers}
    cust_to_idx = {c.node: i for i, c in enumerate(instance.customers)}
    unvisited = list(customers)
    routes, paths, costs, distances, travel_times = [], {}, {}, {}, {}
    
    costs[(instance.depot, instance.depot)] = 0.0
    distances[(instance.depot, instance.depot)] = 0.0
    travel_times[(instance.depot, instance.depot)] = 0.0

    if trace_path_func and precomputed_dicts:
        costs_dict, dists_dict, tt_dict = precomputed_dicts
        def get_dynamic_path(u, v): 
            return None, tt_dict[(u, v)], dists_dict[(u, v)], costs_dict[(u, v)]
    else:
        weighted_graph = add_objective_weights(G, time_weight=time_weight, distance_weight=distance_weight)
        max_speed = _max_edge_speed_mps(G)
        def heuristic_func(n1, n2):
            if method != "astar" or max_speed <= 0: return 0.0
            dist = _straight_line_distance_m(weighted_graph, n1, n2)
            return time_weight * (dist / max_speed) + distance_weight * dist
            
        def get_dynamic_path(u, v):
            try:
                if method == "astar": 
                    path = nx.astar_path(weighted_graph, u, v, heuristic=heuristic_func, weight="_routing_weight")
                else: 
                    path = nx.dijkstra_path(weighted_graph, u, v, weight="_routing_weight")
                tt, dist = path_metrics(weighted_graph, path, weight="_routing_weight")
                return path, tt, dist, time_weight * tt + distance_weight * dist
            except (nx.NetworkXNoPath, nx.NodeNotFound): 
                return None, math.inf, math.inf, math.inf

    def evaluate_route_segment_cost(r):
        total_cost = 0.0
        for u, v in zip(r[:-1], r[1:]):
            if u == v: continue
            _, _, _, obj_cost = get_dynamic_path(u, v)
            if not math.isfinite(obj_cost): return math.inf
            total_cost += obj_cost
        return total_cost

    def apply_2opt(route):
        improved, best_route, best_cost = True, list(route), evaluate_route_segment_cost(route)
        it, max_2opt_iter = 0, 40
        while improved and it < max_2opt_iter:
            improved = False
            it += 1
            n = len(best_route)
            if n <= 3: break
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    new_cost = evaluate_route_segment_cost(new_route)
                    if new_cost < best_cost - 1e-5:
                        best_route, best_cost, improved = new_route, new_cost, True
                        break
                if improved: break
        return best_route

    while unvisited and len(routes) < instance.num_vehicles:
        route = [instance.depot]
        current_load, current_node, current_time = 0.0, instance.depot, 32400.0
        while unvisited:
            best = None
            for candidate in unvisited:
                if current_load + demands[candidate] > instance.vehicle_capacity: continue
                _, tt, dist, obj_cost = get_dynamic_path(current_node, candidate)
                if obj_cost == math.inf: continue
                
                arrival = current_time + tt
                customer = customers[candidate]
                service_start = max(arrival, customer.ready_time)
                waiting = max(0.0, customer.ready_time - arrival)
                lateness = max(0.0, service_start - customer.due_time)

                # Neural prior penalty/bonus injection
                neural_bias = 0.0
                if edge_probs is not None and current_node != instance.depot:
                    u_idx = cust_to_idx[current_node]
                    v_idx = cust_to_idx[candidate]
                    prob = float(edge_probs[u_idx, v_idx])
                    neural_bias = -neural_beta * math.log(max(1e-4, prob))

                candidate_score = (
                    obj_cost 
                    + neural_bias
                    + time_weight * (waiting + customer.service_time) 
                    + (lateness / 60.0) * 10000.0
                )
                
                if best is None or candidate_score < best[0]:
                    best = (candidate_score, candidate, None, tt, dist, obj_cost)
                    
            if best is None: break
            _, candidate, _, tt, dist, obj_cost = best
            route.append(candidate)
            current_load += demands[candidate]
            unvisited.remove(candidate)
            customer = customers[candidate]
            current_time = max(current_time + tt, customer.ready_time) + customer.service_time
            current_node = candidate
        route.append(instance.depot)
        routes.append(apply_2opt(route))

    while len(routes) < instance.num_vehicles: 
        routes.append([instance.depot, instance.depot])

    if unvisited:
        fallback_order = construct_feasible_order(instance)
        if fallback_order is not None:
            fallback_customers = [instance.customers[idx] for idx in fallback_order]
            fallback_routes = split_random_key_solution(fallback_customers, instance)
            if fallback_routes is not None:
                routes = [apply_2opt(r) for r in fallback_routes]
                unvisited = []

    for route in routes:
        for u, v in zip(route[:-1], route[1:]):
            if u == v:
                costs[(u, v)], distances[(u, v)], travel_times[(u, v)] = 0.0, 0.0, 0.0
                continue
            path, tt, dist, obj_cost = get_dynamic_path(u, v)
            costs[(u, v)], distances[(u, v)], travel_times[(u, v)] = obj_cost, dist, tt
            if path is None and trace_path_func:
                paths[(u, v)] = trace_path_func(u, v)
            elif path is not None:
                paths[(u, v)] = path

    score, metrics = evaluate_routes(
        routes, instance, costs, distances, paths,
        time_weight=time_weight, distance_weight=distance_weight, travel_times=travel_times,
    )
    
    if unvisited:
        score += len(unvisited) * 999999.0
        metrics["unserved_customers"] = len(unvisited)

    algo_name = "Neural A*" if edge_probs is not None else ("A* Constructive" if method == "astar" else "Dijkstra Constructive")

    return {
        "algorithm": algo_name,
        "score": score,
        "travel_time_s": metrics.get("travel_time_s", math.inf),
        "distance_m": metrics.get("distance_m", math.inf),
        "service_time_s": metrics.get("service_time_s", 0.0),
        "waiting_time_s": metrics.get("waiting_time_s", 0.0),
        "total_duration_s": metrics.get("total_duration_s", math.inf),
        "tw_penalty": metrics.get("tw_penalty", 0.0),
        "total_lateness_s": metrics.get("total_lateness_s", 0.0),
        "routes": routes, "paths": paths, "history": [score],
    }