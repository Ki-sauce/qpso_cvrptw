import math

import networkx as nx

from .graph_model import add_objective_weights, path_metrics
from .vrp import (
    construct_feasible_order,
    evaluate_routes,
    split_random_key_solution,
)


def _straight_line_distance_m(G, n1, n2):
    """Return a conservative straight-line distance in metres."""
    x1, y1 = G.nodes[n1].get("x"), G.nodes[n1].get("y")
    x2, y2 = G.nodes[n2].get("x"), G.nodes[n2].get("y")
    if None in (x1, y1, x2, y2):
        return 0.0

    crs = G.graph.get("crs")
    is_geographic = getattr(crs, "is_geographic", False)
    if not is_geographic:
        is_geographic = (
            abs(float(x1)) <= 180 and abs(float(x2)) <= 180
            and abs(float(y1)) <= 90 and abs(float(y2)) <= 90
        )
    if is_geographic:
        radius = 6_371_000.0
        lat1, lat2 = math.radians(float(y1)), math.radians(float(y2))
        dlat = lat2 - lat1
        dlon = math.radians(float(x2) - float(x1))
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
    G,
    instance,
    method="dijkstra",
    time_weight=1.0,
    distance_weight=0.0,
):
    """Greedy VRP construction with Dijkstra or admissible A* leg routing.

    The construction remains intentionally greedy, but its final score is
    calculated by the exact same CVRPTW evaluator used by QPSO and GA.
    """
    weighted_graph = add_objective_weights(
        G, time_weight=time_weight, distance_weight=distance_weight
    )
    demands = {c.node: c.demand for c in instance.customers}
    customers = {c.node: c for c in instance.customers}
    unvisited = list(customers)
    routes = []
    paths = {}
    costs = {}
    distances = {}
    travel_times = {}
    costs[(instance.depot, instance.depot)] = 0.0
    distances[(instance.depot, instance.depot)] = 0.0
    travel_times[(instance.depot, instance.depot)] = 0.0
    max_speed = _max_edge_speed_mps(G)

    def heuristic_func(n1, n2):
        if method != "astar" or max_speed <= 0:
            return 0.0
        distance = _straight_line_distance_m(weighted_graph, n1, n2)
        lower_bound_time = distance / max_speed
        return time_weight * lower_bound_time + distance_weight * distance

    def get_dynamic_path(u, v):
        try:
            if method == "astar":
                path = nx.astar_path(
                    weighted_graph,
                    u,
                    v,
                    heuristic=heuristic_func,
                    weight="_routing_weight",
                )
            else:
                path = nx.dijkstra_path(
                    weighted_graph, u, v, weight="_routing_weight"
                )
            travel_time, distance = path_metrics(
                weighted_graph, path, weight="_routing_weight"
            )
            objective_cost = time_weight * travel_time + distance_weight * distance
            return path, travel_time, distance, objective_cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, math.inf, math.inf, math.inf

    while unvisited and len(routes) < instance.num_vehicles:
        route = [instance.depot]
        current_load = 0.0
        current_node = instance.depot
        current_time = 32400.0

        while unvisited:
            best = None
            for candidate in unvisited:
                if current_load + demands[candidate] > instance.vehicle_capacity:
                    continue
                path, travel_time, distance, objective_cost = get_dynamic_path(
                    current_node, candidate
                )
                if path is None:
                    continue

                arrival = current_time + travel_time
                customer = customers[candidate]
                service_start = max(arrival, customer.ready_time)
                waiting = max(0.0, customer.ready_time - arrival)
                lateness = max(0.0, service_start - customer.due_time)
                candidate_score = (
                    objective_cost
                    + time_weight * (waiting + customer.service_time)
                    + (lateness / 60.0) * 10000.0
                )
                if best is None or candidate_score < best[0]:
                    best = (
                        candidate_score,
                        candidate,
                        path,
                        travel_time,
                        distance,
                        objective_cost,
                    )

            if best is None:
                break

            _, candidate, path, travel_time, distance, objective_cost = best
            route.append(candidate)
            paths[(current_node, candidate)] = path
            costs[(current_node, candidate)] = objective_cost
            distances[(current_node, candidate)] = distance
            travel_times[(current_node, candidate)] = travel_time
            current_load += demands[candidate]
            unvisited.remove(candidate)
            arrival = current_time + travel_time
            customer = customers[candidate]
            current_time = max(arrival, customer.ready_time) + customer.service_time
            current_node = candidate

        path, travel_time, distance, objective_cost = get_dynamic_path(
            current_node, instance.depot
        )
        route.append(instance.depot)
        if path is not None:
            paths[(current_node, instance.depot)] = path
            costs[(current_node, instance.depot)] = objective_cost
            distances[(current_node, instance.depot)] = distance
            travel_times[(current_node, instance.depot)] = travel_time
        routes.append(route)

    while len(routes) < instance.num_vehicles:
        routes.append([instance.depot, instance.depot])

    if unvisited:
        # The greedy construction is allowed to choose a poor bin-packing
        # order, but it must never make the baseline appear to serve a
        # different customer set.  Repair the order with a capacity-feasible
        # assignment and then rebuild every leg using the same A*/Dijkstra
        # router.  This preserves the baseline's routing method while making
        # its output comparable with GA and QPSO.
        fallback_order = construct_feasible_order(instance)
        if fallback_order is not None:
            fallback_customers = [instance.customers[index] for index in fallback_order]
            fallback_routes = split_random_key_solution(fallback_customers, instance)
            if fallback_routes is not None:
                routes = fallback_routes
                paths = {}
                costs = {}
                distances = {}
                travel_times = {}
                costs[(instance.depot, instance.depot)] = 0.0
                distances[(instance.depot, instance.depot)] = 0.0
                travel_times[(instance.depot, instance.depot)] = 0.0

                for route in routes:
                    for u, v in zip(route[:-1], route[1:]):
                        if u == v:
                            costs[(u, v)] = 0.0
                            distances[(u, v)] = 0.0
                            travel_times[(u, v)] = 0.0
                            continue
                        path, travel_time, distance, objective_cost = get_dynamic_path(u, v)
                        costs[(u, v)] = objective_cost
                        distances[(u, v)] = distance
                        travel_times[(u, v)] = travel_time
                        if path is not None:
                            paths[(u, v)] = path
                unvisited = []

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
    if unvisited:
        score += len(unvisited) * 999999.0
        metrics["unserved_customers"] = len(unvisited)
        metrics["objective"] = score

    name = "A* Constructive" if method == "astar" else "Dijkstra Constructive"
    return {
        "algorithm": name,
        "score": score,
        "travel_time_s": metrics.get("travel_time_s", math.inf),
        "distance_m": metrics.get("distance_m", math.inf),
        "service_time_s": metrics.get("service_time_s", 0.0),
        "waiting_time_s": metrics.get("waiting_time_s", 0.0),
        "total_duration_s": metrics.get("total_duration_s", math.inf),
        "tw_penalty": metrics.get("tw_penalty", 0.0),
        "total_lateness_s": metrics.get("total_lateness_s", 0.0),
        "routes": routes,
        "paths": paths,
        "history": [score],
    }
